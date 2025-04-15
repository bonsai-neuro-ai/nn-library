import torch


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
    def __init__(
        self,
        n_x: int,
        n_y: int,
        bias: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.n = 0
        self.n_x = n_x
        self.n_y = n_y
        self.bias = bias
        self.mean_x = torch.zeros(n_x, device=device)
        self.mean_y = torch.zeros(n_y, device=device)
        self.xtx = torch.zeros((n_x, n_x), device=device)
        self.xty = torch.zeros((n_x, n_y), device=device)

    def reset(self) -> None:
        self.n = 0
        self.mean_x.zero_()
        self.mean_y.zero_()
        self.xtx.zero_()
        self.xty.zero_()

    @torch.no_grad()
    def add_batch(self, from_data: torch.Tensor, to_data: torch.Tensor) -> None:
        batch_size = from_data.size(0)
        self.n += batch_size

        # Update running means only if bias is enabled (otherwise, behaves as if means=0)
        if self.bias:
            self.mean_x += (from_data.sum(dim=0) - batch_size * self.mean_x) / self.n
            self.mean_y += (to_data.sum(dim=0) - batch_size * self.mean_y) / self.n

        # Update running-average products A^T A and A^T B
        batch_a = torch.einsum("bi,bj->ij", from_data, from_data)
        batch_b = torch.einsum("bi,bj->ij", from_data, to_data)
        self.xtx += (batch_a - batch_size * self.xtx) / self.n
        self.xty += (batch_b - batch_size * self.xty) / self.n

    @torch.no_grad()
    def solve(self, ridge: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate weights w and bias b from batch statistics added so far."""
        if self.bias:
            ata = self.xtx - self.mean_x[:, None] @ self.mean_x[None, :]
            atb = self.xty - self.mean_x[:, None] @ self.mean_y[None, :]
        else:
            ata = self.xtx
            atb = self.xty

        w = safe_linalg_lstsq(ata + ridge * torch.eye(self.n_x, device=ata.device), atb)
        b = self.mean_y - self.mean_x @ w

        return w, b
