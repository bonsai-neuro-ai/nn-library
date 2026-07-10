from collections import defaultdict
from typing import Iterable, Optional

import torch
from typing_extensions import deprecated


class RunningAverage[T]:
    """
    Incrementally tracks a weighted running mean of values seen in batches, without needing to
    keep all the data in memory. `T` is typically a `torch.Tensor` but can be anything supporting
    `+`, `-`, and scalar multiplication/division (e.g. a plain float).
    """

    def __init__(self):
        self.avg: T | None = None
        self.count: int = 0

    def update(self, batch_avg: T, batch_count: int):
        """
        Fold in a new batch's average.

        :param batch_avg: the mean value of the new batch (same type/shape as previous batches).
        :param batch_count: number of samples the new batch's average was computed over, used to
            weight its contribution to the running mean.
        """
        if self.avg is None:
            self.avg = batch_avg
        else:
            self.avg = self.avg + (batch_avg - self.avg) * batch_count / (self.count + batch_count)
        self.count += batch_count


class RunningCovariance:
    """
    Welford algorithm to track running variance or covariance or cross-covariance (more
    numerically stable than RunningAverage(x*y) - RunningAverage(x)*RunningAverage(y)). Note that
    where RunningAverage accepts a precomputed average of each batch of values, this class
    expects the raw values.

    Whereas we could write RunningAverage in a type-agnostic way, this class assumes torch Tensors.
    It is otherwise too awkward to handle initialization, copies, updates, etc. with generic types.

    This algorithm assumes by default the mean is unknown ahead of time and hence the dof=1
    correction is the default. If the mean is known, you're better off using RunningAverage and
    feeding it (values-known_mean)**2. Or if you prefer the biased 1/N variance calculation,
    set dof=0.

    For variance of sets of scalars, call update(x). For a covariance matrix, call update(x, x). For
    a cross-covariance matrix, call update(x, y).

    Note that 'RunningCovariance' has an .avg property and a .count property, so it duck-types just
    like a RunningAverage instance.

    :param dof: Degrees of freedom. default 1. When 'variance' is returned, it is normalized by
        1/(N-dof).
    :param scalar: whether we are tracking the per-element variances or the element-by-element
        covariances. Default False (covariances).
    """

    def __init__(self, dof: int = 1, scalar: bool = False):
        self.count: int = 0
        self.mu_x: torch.Tensor | None = None
        self.mu_y: torch.Tensor | None = None
        self._sum_of_squares: torch.Tensor | None = None
        self._dof = dof
        self._scalar = scalar

    def _update_scalar(self, values_x: Iterable[torch.Tensor]):
        for val_x in values_x:
            if self.count == 0:
                self.mu_x = torch.clone(val_x)
                self._sum_of_squares = torch.zeros_like(val_x)
                self.count = 1
            else:
                # Running updates (see Welford's incremental algorithm in
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).
                self.count += 1
                new_mu_x = self.mu_x + (val_x - self.mu_x) / self.count
                self._sum_of_squares += (val_x - new_mu_x) * (val_x - self.mu_x)
                self.mu_x = new_mu_x

    def _update_vector(self, values_x: Iterable[torch.Tensor], values_y: Iterable[torch.Tensor]):
        for val_x, val_y in zip(values_x, values_y):
            val_x, val_y = val_x.flatten(), val_y.flatten()
            if self.count == 0:
                self.mu_x = torch.clone(val_x)
                self.mu_y = torch.clone(val_y)
                self._sum_of_squares = torch.zeros(
                    (len(val_x), len(val_y)), dtype=val_x.dtype, device=val_x.device
                )
                self.count = 1
            else:
                # Running updates (see the online covariance algorithm in
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance). Note it's
                # important that the update uses the 'old' mu_x and the 'new' mu_y,
                # so the updates are staggered.
                self.count += 1
                self.mu_y = self.mu_y + (val_y - self.mu_y) / self.count
                self._sum_of_squares += (val_x - self.mu_x)[:, None] * (val_y - self.mu_y)[None, :]
                self.mu_x = self.mu_x + (val_x - self.mu_x) / self.count

    def update(
        self, values_x: Iterable[torch.Tensor], values_y: Optional[Iterable[torch.Tensor]] = None
    ):
        """
        Fold in a new batch of values. Unlike RunningAverage, this should be a raw set of values.
        In other words, the caller should not precompute any batch means or batch variances.

        :param values_x: an iterable of tensor values. If scalar=True and values_x has shape (
        batch, a, b, c) then we keep track of an (a, b, c) shaped tensor of variances of each
        element. If scalar=False and values_x is the only input, it is flattened, and we keep track
        of a (a*b*c, a*b*c) shaped covariance matrix.
        :param values_y: only valid if scalar=False. If provided, values_y defines the columns of
        the cross-covariance matrix where values_x defines the rows.
        """
        if self._scalar:
            if values_y is not None:
                raise ValueError("Do not pass in values_y if calculating scalar variances")
            self._update_scalar(values_x)
        else:
            if values_y is None:
                values_y = values_x
            self._update_vector(values_x, values_y)

    @property
    def variance(self) -> torch.Tensor:
        if not self._scalar:
            raise ValueError("Call 'covariance' for the non-scalar case")
        if self._sum_of_squares is None or self.count < self._dof:
            raise ValueError("No data yet; call update() with some values first")
        return self._sum_of_squares / (self.count - self._dof)

    @property
    def covariance(self) -> torch.Tensor:
        if self._scalar:
            raise ValueError("Call 'variance' for the scalar case")
        if self._sum_of_squares is None or self.count < self._dof:
            raise ValueError("No data yet; call update() with some values first")
        return self._sum_of_squares / (self.count - self._dof)

    @property
    def avg(self):
        if self._scalar:
            return self.variance
        else:
            return self.covariance


def calculate_moments_batchwise(
    batches: Iterable[torch.Tensor] | Iterable[tuple[torch.Tensor] | list[torch.Tensor]],
    covariances: bool = False,
) -> dict[str, RunningAverage]:
    """
    Stream over batches of one or more aligned tensors and accumulate first and second moments
    (means and cross-covariance-like products) for each, without materializing the full dataset.

    Each item yielded by `batches` is a tuple of tensors of shape (batch, ...) representing
    aligned variables (e.g. activations from different layers/models on the same inputs). Each
    tensor is flattened to (batch, features) before accumulating.

    :param batches: iterable of tuples of same-length tensors, one tuple per minibatch.
    :return: dict mapping "moment1_{i}" -> mean of variable i, and "moment2_{i}_{j}" (for
        i <= j) -> average product of x_i[:, None] * x_j[None, :] / batch_size, the uncentered
        second moment. If 'covariances' is set to True, also calculates "cov_{i}_{j}" terms using
        the numerically stable Welford algorithm.
    """
    moments = {}

    def _init_moments(num_tensors):
        for j in range(num_tensors):
            moments[f"moment1_{j}"] = RunningAverage()
            for i in range(j + 1):
                moments[f"moment2_{i}_{j}"] = RunningAverage()
                if covariances:
                    moments[f"cov_{i}_{j}"] = RunningCovariance(scalar=False)

    for batch in batches:
        if torch.is_tensor(batch):
            batch = [batch]

        if not moments:
            _init_moments(len(batch))

        for i, x in enumerate(batch):
            x = x.flatten(start_dim=1)
            m, n_x = x.shape
            moments[f"moment1_{i}"].update(torch.mean(x, dim=0), m)
            for j, y in enumerate(batch):
                if i > j:
                    continue
                y = y.flatten(start_dim=1)
                moments[f"moment2_{i}_{j}"].update(torch.einsum("mi,mj->ij", x, y) / m, m)
                if covariances:
                    moments[f"cov_{i}_{j}"].update(x, y)

    return dict(moments)


@deprecated("Use RunningCovariance or call calculate_moments_batchwise(covariance=True) instead")
def moments_to_covs[T](moments: dict[str, RunningAverage[T]], centered: bool) -> dict[str, T]:
    """
    Convert the moments dict produced by `calculate_moments_batchwise` into (co)variance matrices.

    :param moments: output of `calculate_moments_batchwise`.
    :param centered: if True, subtract the outer product of means to get true covariance matrices
        (i.e. center the data); if False, return the raw uncentered second moments.
    :return: dict mapping "cov_{i}_{j}" -> covariance (or uncentered second moment) matrix between
        variables i and j.
    """
    out = {}
    for k, v in moments.items():
        if k.startswith("moment2"):
            _, i, j = k.split("_")
            i, j = int(i), int(j)
            if centered:
                moment1_i = moments[f"moment1_{i}"].avg
                moment1_j = moments[f"moment1_{j}"].avg
                out[f"cov_{i}_{j}"] = v.avg - moment1_i[:, None] * moment1_j[None, :]
            else:
                out[f"cov_{i}_{j}"] = v.avg

    return out


__all__ = [
    "RunningAverage",
    "RunningCovariance",
    "calculate_moments_batchwise",
]
