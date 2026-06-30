import torch

from nn_lib.utils import RunningAverage, eye_like


def safe_linalg_lstsq(
    a: torch.Tensor, b: torch.Tensor, symmetric: bool = False, rcond=1e-6, eps=1e-15
) -> torch.Tensor:
    """Wrapper around torch.linalg.lstsq but handles singular inputs on non-CPU devices, where
    torch.linalg.lstsq fails silently. See https://github.com/pytorch/pytorch/issues/117122
    """
    if a.device == torch.device("cpu"):
        return torch.linalg.lstsq(a, b).solution
    else:
        if symmetric:
            s, u = torch.linalg.eigh(a)
            vh = u.T
        else:
            u, s, vh = torch.linalg.svd(a, full_matrices=False)

        s_pseudo_inverse = 1 / (s + eps)
        s_pseudo_inverse[s < rcond * s.max()] = 0

        return vh.T @ (s_pseudo_inverse[:, None] * (u.T @ b))


def safe_regression(
    x: torch.Tensor, y: torch.Tensor, bias: bool, ridge: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Y ≈ XW + B for W and B using least squares regression and an optional ridge penalty.

    Args:
        x (torch.Tensor): Input data of shape (n_samples, n_features).
        y (torch.Tensor): Target data of shape (n_samples, n_targets).
        bias (bool): Whether to include a bias term. If False, returns zeros for bias.
        ridge (float): Ridge penalty for regularization. Default is 0.0 (no regularization).
    """
    m, n_x = x.shape
    _, n_y = y.shape

    if bias:
        m_x, m_y = x.mean(dim=0), y.mean(dim=0)
        x, y = x - m_x, y - m_y
    else:
        m_x, m_y = x.new_zeros(n_x), y.new_zeros(n_y)

    a = torch.einsum("bi,bj->ij", x, x) / m
    b = torch.einsum("bi,bj->ij", x, y) / m

    w = safe_linalg_lstsq(a + ridge * torch.eye(n_x, device=a.device), b)
    b = m_y - m_x @ w

    return w, b


class StreamingLinearRegression(object):
    """
    Accumulates sufficient statistics (means, X^T X, X^T Y) for linear regression across
    multiple minibatches, then solves the normal equations once all data has been seen.

    This lets you fit a linear regression on a dataset that is too large to hold in memory at
    once. Typical usage::

        slr = StreamingLinearRegression()
        for x_batch, y_batch in dataloader:
            slr.add_batch(x_batch.flatten(1), y_batch.flatten(1))
        w, b = slr.solve(bias=True, ridge=1e-3)

    See also `safe_regression` for an in-memory version that works on a single batch.
    """

    def __init__(self):
        self._mean_x: RunningAverage[torch.Tensor] = RunningAverage()
        self._mean_y: RunningAverage[torch.Tensor] = RunningAverage()
        self._xtx: RunningAverage[torch.Tensor] = RunningAverage()
        self._xty: RunningAverage[torch.Tensor] = RunningAverage()

    @property
    def mean_x(self) -> torch.Tensor:
        if self._mean_x.avg is None:
            raise ValueError("No batches have been added yet. Call add_batch() first.")
        return self._mean_x.avg

    @property
    def mean_y(self) -> torch.Tensor:
        if self._mean_y.avg is None:
            raise ValueError("No batches have been added yet. Call add_batch() first.")
        return self._mean_y.avg

    @property
    def xtx(self) -> torch.Tensor:
        if self._xtx.avg is None:
            raise ValueError("No batches have been added yet. Call add_batch() first.")
        return self._xtx.avg

    @property
    def xty(self) -> torch.Tensor:
        if self._xty.avg is None:
            raise ValueError("No batches have been added yet. Call add_batch() first.")
        return self._xty.avg

    @torch.no_grad()
    def add_batch(self, from_data: torch.Tensor, to_data: torch.Tensor) -> None:
        """
        Accumulate statistics for one batch of (input, output) pairs.

        :param from_data: input data of shape (batch, n_features).
        :param to_data: target data of shape (batch, n_targets).
        """
        batch_size = from_data.size(0)
        self._mean_x.update(torch.mean(from_data, dim=0), batch_size)
        self._mean_y.update(torch.mean(to_data, dim=0), batch_size)
        self._xtx.update(torch.einsum("bi,bj->ij", from_data, from_data) / batch_size, batch_size)
        self._xty.update(torch.einsum("bi,bj->ij", from_data, to_data) / batch_size, batch_size)

    @torch.no_grad()
    def solve(self, bias: bool = True, ridge: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the linear regression using the accumulated statistics.

        :param bias: if True, fit an intercept by centering the data (so the intercept is
            estimated as mean_y - mean_x @ w).
        :param ridge: L2 regularization strength added to the diagonal of X^T X before solving.
        :return: (w, b) where w has shape (n_features, n_targets) and b has shape (n_targets,).
        """
        if bias:
            ata = self.xtx - self.mean_x[:, None] @ self.mean_x[None, :]
            atb = self.xty - self.mean_x[:, None] @ self.mean_y[None, :]

            w = safe_linalg_lstsq(ata + ridge * eye_like(ata), atb)
            b = self.mean_y - self.mean_x @ w
        else:
            ata = self.xtx
            atb = self.xty

            w = safe_linalg_lstsq(ata + ridge * eye_like(ata), atb)
            b = torch.zeros_like(self.mean_y)

        return w, b
