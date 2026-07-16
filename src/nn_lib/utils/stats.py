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


class RunningVariance(object):
    """
    Welford algorithm to track running variance (more numerically stable than RunningAverage(x*y)
    - RunningAverage(x)*RunningAverage(y)). Note that where RunningAverage accepts a precomputed
    average of each batch of values, this class expects the raw values.

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

    @staticmethod
    def _get_or_init_mean(
        count: int, centered: bool, mu: torch.Tensor | None, val: torch.Tensor
    ) -> torch.Tensor:
        """Instantiate or calculate new means. Does not write to self._mu."""
        if count <= 1:
            if centered:
                return torch.clone(val)
            else:
                # Allocate new zeros at initialization matching the specs of val. The 'uncentered'
                # case is equivalent to setting means equal to zero.
                return torch.zeros_like(val)
        else:
            if centered:
                new_mu = mu + (val - mu) / count
                return new_mu
            else:
                # Whenever count > 0, we can assume mu was initialized to zeros. Just re-return the
                # same zeros rather than allocating new zeros.
                return mu

    def update(self, values_x: Iterable[torch.Tensor]):
        """
        Fold in a new batch of values. Unlike RunningAverage, this should be a raw set of values.
        In other words, the caller should not precompute any batch means or batch variances.

        :param values_x: an iterable batch of tensor values.
        """
        for x in values_x:
            self.count += 1
            new_mu_x = self._get_or_init_mean(self.count, self.centered, self._mu_x, x)
            if self.count == 1:
                self._sum_of_squares = torch.zeros_like(x) if self.centered else x**2
            else:
                # Running updates (see Welford's incremental algorithm in
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).
                self._sum_of_squares += (x - new_mu_x) * (x - self._mu_x)
            self._mu_x = new_mu_x

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

    @staticmethod
    def _get_or_init_mean(
        count: int, centered: bool, mu: torch.Tensor | None, val: torch.Tensor
    ) -> torch.Tensor:
        """Instantiate or calculate new means. Does not write to self._mu_x or self._mu_y."""
        if count <= 1:
            if centered:
                return torch.clone(val)
            else:
                # Allocate new zeros at initialization matching the specs of val. The 'uncentered'
                # case is equivalent to setting means equal to zero.
                return torch.zeros_like(val)
        else:
            if centered:
                new_mu = mu + (val - mu) / count
                return new_mu
            else:
                # Whenever count > 0, we can assume mu was initialized to zeros. Just re-return the
                # same zeros rather than allocating new zeros.
                return mu

    def update(
        self, values_x: Iterable[torch.Tensor], values_y: Optional[Iterable[torch.Tensor]] = None
    ):
        """
        Fold in a new batch of values. Unlike RunningAverage, this should be a raw set of values.
        In other words, the caller should not precompute any batch means or batch variances.

        :param values_x: an iterable of tensor values. If values_x has shape (batch, a, b,
        c) then  it is flattened, and we keep track of a (a*b*c, a*b*c) shaped covariance matrix.
        :param values_y: If provided, values_y defines the columns of the cross-covariance matrix
        where values_x defines the rows.
        """
        if values_y is None:
            values_y = values_x

        for x, y in zip(values_x, values_y):
            self.count += 1
            x, y = x.flatten(), y.flatten()
            new_mu_x = self._get_or_init_mean(self.count, self.centered, self._mu_x, x)
            new_mu_y = self._get_or_init_mean(self.count, self.centered, self._mu_y, y)
            if self.count == 1:
                self._sum_of_squares = (
                    torch.zeros((len(x), len(y)), dtype=x.dtype, device=x.device)
                    if self.centered
                    else x[:, None] * y[None, :]
                )
            else:
                # Running updates (see the online covariance algorithm in
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance). Note it's
                # important that the update uses the 'old' mu_x and the 'new' mu_y (or vice versa).
                self._sum_of_squares += (x - self._mu_x)[:, None] * (y - new_mu_y)[None, :]
            self._mu_x = new_mu_x
            self._mu_y = new_mu_y

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
    "RunningVariance",
    "RunningCovariance",
]
