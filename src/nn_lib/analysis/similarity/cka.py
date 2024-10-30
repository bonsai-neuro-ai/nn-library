import torch
from enum import Enum, auto


class HSICEstimator(Enum):
    """Estimators for the Hilbert-Schmidt Independence Criterion (HSIC)"""

    GRETTON2005 = auto()
    SONG2007 = auto()
    LANGE2022 = auto()


def center(k: torch.Tensor):
    """Double-center the given m by m kernel matrix K"""
    assert k.dim() == 2, "Input tensor must be 2D"
    assert k.size(0) == k.size(1), "Input tensor must be square"
    m = k.size(0)
    h = torch.eye(m) - torch.ones(m, m) / m
    return h @ k @ h


def remove_diagonal(k: torch.Tensor):
    """Remove the diagonal from the given m by m kernel matrix K (set it to zero)"""
    assert k.dim() == 2, "Input tensor must be 2D"
    assert k.size(0) == k.size(1), "Input tensor must be square"
    m = k.size(0)
    return k * (1 - torch.eye(m, device=k.device, dtype=k.dtype))


# TODO - allow for nonlinear kernels
def hsic(x: torch.Tensor, y: torch.Tensor, estimator: HSICEstimator = HSICEstimator.GRETTON2005):
    """Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two sets of samples"""
    m = x.size(0)
    assert m == y.size(0), "Input tensors must have the same # rows"

    # Compute the kernel matrices
    k_x = torch.einsum("i...,j...->ij", x, x)
    k_y = torch.einsum("i...,j...->ij", y, y)

    match estimator:
        case HSICEstimator.GRETTON2005:
            # Compute the HSIC using the estimator from Gretton et al. (2005)
            return torch.sum(center(k_x) * center(k_y)) / (m - 1) ** 2
        case HSICEstimator.SONG2007:
            # Compute the HSIC using the estimator from Song et al. (2007)
            k_x, k_y = remove_diagonal(k_x), remove_diagonal(k_y)
            return (
                (k_x * k_y).sum()
                - 2 * (k_x.sum(dim=0) * k_y.sum(dim=0)).sum() / (m - 2)
                + k_x.sum() * k_y.sum() / ((m - 1) * (m - 2))
            ) / (m * (m - 3))
        case HSICEstimator.LANGE2022:
            # Compute the HSIC using the estimator from Lange et al. (2022)
            k_x, k_y = remove_diagonal(k_x), remove_diagonal(k_y)
            return torch.sum(k_x * k_y) / (m * (m - 3))


# TODO - allow for nonlinear kernels
def cka(x: torch.Tensor, y: torch.Tensor, estimator: HSICEstimator = HSICEstimator.GRETTON2005):
    """Compute the Centered Kernel Alignment (CKA) between two sets of samples"""
    hsic_xy = hsic(x, y, estimator)
    hsic_xx = hsic(x, x, estimator)
    hsic_yy = hsic(y, y, estimator)
    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)


__all__ = ["HSICEstimator", "hsic", "cka"]
