from typing import Iterable, Optional

import torch


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


def _as_batch(values: Iterable[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Return values as a single tensor whose leading dimension is the batch dimension."""
    if isinstance(values, torch.Tensor):
        return values
    values = list(values)
    if len(values) == 0:
        return torch.empty(0)
    return torch.stack(values)


class RunningVariance:
    """
    Tracks running (elementwise) variance using batched Welford / Chan et al. pairwise updates
    (more numerically stable than RunningAverage(x*y) - RunningAverage(x)*RunningAverage(y)).
    Note that where RunningAverage accepts a precomputed average of each batch of values, this
    class expects the raw values.

    Whereas we could write RunningAverage in a type-agnostic way, this class assumes torch Tensors.
    It is otherwise too awkward to handle initialization, copies, updates, etc. with generic types.

    This algorithm assumes by default the mean is unknown ahead of time and hence the dof=1
    correction is the default, i.e. querying the resulting variances gives the unbiased estimate

    Note that 'RunningVariance' has an .avg property and a .count property, so it duck-types just
    like a RunningAverage instance.

    :param centered: Whether to track empirical means. If False, assumes mu_x=mu_y=zeros and dof=0.
        Default is True, so the resulting variances and covariances match the defaults for np.var
        or torch.var.
    """

    def __init__(self, centered: bool = True):
        self.count: int = 0
        self._mu_x: torch.Tensor | None = None
        self.centered = centered
        self._sum_of_squares: torch.Tensor | None = None
        self._dof = 1 if centered else 0

    def update(self, values_x: Iterable[torch.Tensor] | torch.Tensor):
        """
        Fold in a new batch of values. Unlike RunningAverage, this should be a raw set of values.
        In other words, the caller should not precompute any batch means or batch variances.

        :param values_x: a batch of values, either a tensor whose first dimension is the batch
            dimension (fastest) or an iterable of same-shaped tensors (stacked internally).
        """
        x = _as_batch(values_x)
        b = x.shape[0]
        if b == 0:
            return
        m = self.count

        if self.centered:
            # Per-batch mean and sum of squared deviations
            mu_b = x.mean(dim=0)
            dx = x - mu_b
            s_b = (dx * dx).sum(dim=0)
            if m == 0:
                self._mu_x, self._sum_of_squares = mu_b, s_b
            else:
                # Pairwise merge: S^{m+b} = S^m + S^b + mb/(m+b) * (mu^b - mu^m)^2
                delta = mu_b - self._mu_x
                self._sum_of_squares = (
                    self._sum_of_squares + s_b + delta * delta * (m * b / (m + b))
                )
                self._mu_x = self._mu_x + delta * (b / (m + b))
        else:
            s_b = (x * x).sum(dim=0)
            if m == 0:
                self._mu_x, self._sum_of_squares = torch.zeros_like(x[0]), s_b
            else:
                self._sum_of_squares = self._sum_of_squares + s_b

        self.count = m + b

    @property
    def mu_x(self) -> torch.Tensor:
        if self._mu_x is None:
            raise ValueError("No data yet; call update() with some values first")
        return self._mu_x

    @property
    def variance(self) -> torch.Tensor:
        if self._sum_of_squares is None or self.count < self._dof:
            raise ValueError("No data yet; call update() with some values first")
        return self._sum_of_squares / (self.count - self._dof)

    @property
    def avg(self):
        return self.variance


class RunningCovariance:
    """Like RunningVariance but tracks covariance matrices instead of scalar variances.

    Uses the batched pairwise update of Chan et al. (1983), which reduces to Welford's algorithm
    for batches of size 1:

        S^{m+b} = S^m + S^b + mb/(m+b) * (mu_x^b - mu_x^m)(mu_y^b - mu_y^m)^T

    :param centered: Whether to track empirical means. If False, assumes mu_x=mu_y=zeros and dof=0.
        Default is True, so the resulting variances and covariances match the defaults for np.var
        or torch.var.
    """

    def __init__(self, centered: bool = True):
        self.count: int = 0
        self._mu_x: torch.Tensor | None = None
        self._mu_y: torch.Tensor | None = None
        self.centered = centered
        self._sum_of_squares: torch.Tensor | None = None
        self._dof = 1 if centered else 0

    def update(
        self,
        values_x: Iterable[torch.Tensor] | torch.Tensor,
        values_y: Optional[Iterable[torch.Tensor] | torch.Tensor] = None,
    ):
        """
        Fold in a new batch of values. Unlike RunningAverage, this should be a raw set of values.
        In other words, the caller should not precompute any batch means or batch variances.

        :param values_x: a batch of values, either a tensor whose first dimension is the batch
            dimension (fastest) or an iterable of same-shaped tensors (stacked internally). If
            values_x has shape (batch, a, b, c) then each item is flattened, and we keep track of
            a (a*b*c, a*b*c) shaped covariance matrix.
        :param values_y: If provided, values_y defines the columns of the cross-covariance matrix
            where values_x defines the rows. Must have the same batch size as values_x.
        """
        x = _as_batch(values_x)
        b = x.shape[0]
        if b == 0:
            return
        x = x.reshape(b, -1)
        if values_y is None:
            y = x
        else:
            y = _as_batch(values_y).reshape(b, -1)
            if y.shape[0] != b:
                raise ValueError(f"Batch sizes differ: x has {b}, y has {y.shape[0]}")
        m = self.count

        if self.centered:
            # Per-batch means and sum of outer products of deviations, S^b
            mu_x_b = x.mean(dim=0)
            dx = x - mu_x_b
            if values_y is None:
                mu_y_b, dy = mu_x_b, dx
            else:
                mu_y_b = y.mean(dim=0)
                dy = y - mu_y_b
            s_b = dx.T @ dy

            if m == 0:
                self._mu_x, self._mu_y, self._sum_of_squares = mu_x_b, mu_y_b, s_b
            else:
                # Pairwise merge of (old running stats) with (new batch stats)
                delta_x = mu_x_b - self._mu_x
                delta_y = mu_y_b - self._mu_y
                correction = torch.outer(delta_x, delta_y) * (m * b / (m + b))
                self._sum_of_squares = self._sum_of_squares + s_b + correction
                self._mu_x = self._mu_x + delta_x * (b / (m + b))
                self._mu_y = self._mu_y + delta_y * (b / (m + b))
        else:
            s_b = x.T @ y
            if m == 0:
                self._mu_x = torch.zeros_like(x[0])
                self._mu_y = torch.zeros_like(y[0])
                self._sum_of_squares = s_b
            else:
                self._sum_of_squares = self._sum_of_squares + s_b

        self.count = m + b

    @property
    def mu_x(self) -> torch.Tensor:
        if self._mu_x is None:
            raise ValueError("No data yet; call update() with some values first")
        return self._mu_x

    @property
    def mu_y(self) -> torch.Tensor:
        if self._mu_y is None:
            raise ValueError("No data yet; call update() with some values first")
        return self._mu_y

    @property
    def covariance(self) -> torch.Tensor:
        if self._sum_of_squares is None or self.count < self._dof:
            raise ValueError("No data yet; call update() with some values first")
        return self._sum_of_squares / (self.count - self._dof)

    @property
    def avg(self):
        return self.covariance


__all__ = [
    "RunningAverage",
    "RunningCovariance",
    "RunningVariance",
]
