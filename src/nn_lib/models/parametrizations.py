"""Custom additions to PyTorch's parametrization utilities; see
https://pytorch.org/tutorials/intermediate/parametrizations.html
"""

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization


class LowRankParametrization(nn.Module):
    """A parametrization module using SVD decomposition of a weight matrix to enforce low-rank."""

    # TODO - do we need to enforce that u and vh are orthogonal?

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def forward(self, u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor):
        return u[:, : self.rank] @ torch.diag_embed(s[: self.rank]) @ vh[: self.rank, :]

    @torch.no_grad()
    def right_inverse(self, weight: torch.Tensor):
        u, s, vh = torch.linalg.svd(weight)
        return u[:, : self.rank], s[: self.rank], vh[: self.rank, :]


def low_rank(module: nn.Module, name: str = "weight", rank: Optional[int] = None) -> nn.Module:
    """Register a low-rank parametrization for a weight matrix in a module."""
    if rank is None:
        rank = min(*getattr(module, name).shape)
    parametrization = LowRankParametrization(rank)
    register_parametrization(module, name, parametrization, unsafe=True)
    return module


class OrthogonalParametrization(nn.Module):
    """Custom parametrization enforcing orthonormal columns in a weight matrix. Taking inspiration
    from torch.nn.utils.parametrizations.orthogonal but simplifying quite a bit.

    Parametrizes the space of orthogonal matrices using the tangent space at a base point. The base
    point is an orthogonal matrix, and the tangent space is the set of skew-symmetric matrices.
    """

    base: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("base", None)

    def forward(self, tangent_vector: torch.Tensor):
        # Ensure the tangent vector is skew-symmetric and take the exponential map to get an
        # orthogonal matrix.
        tangent_vector = (tangent_vector - tangent_vector.transpose(-1, -2)) / 2
        rot = torch.linalg.matrix_exp(tangent_vector)

        rows, cols = self.base.shape[-2:]
        if rows <= cols:
            return rot @ self.base
        else:
            return self.base @ rot

    @torch.no_grad()
    def right_inverse(self, weight: torch.Tensor):
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        self.base = u @ vh

        # With the base point updated, the tangent vector for 'weight' is all zeros. We're at that
        # point in the manifold already. If weight has fewer rows than cols, u will be smaller than
        # vh and the tangent space will be the set of skew-symmetric matrices of the same size as u.
        # Vice versa if vh is smaller than u.
        rows, cols = weight.shape[-2:]
        return torch.zeros_like(u) if rows <= cols else torch.zeros_like(vh)


def orthogonal(module: nn.Module, name: str = "weight"):
    """Register an orthogonal parametrization for a weight matrix in a module."""
    orth = OrthogonalParametrization()
    register_parametrization(module, name, orth, unsafe=True)
    return module


class ScaledOrthogonalParametrization(OrthogonalParametrization):
    def forward(self, tangent_vector: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return super().forward(tangent_vector) * scale

    @torch.no_grad()
    def right_inverse(self, weight: torch.Tensor):
        _, s, _ = torch.linalg.svd(weight)
        rms_scale = torch.sqrt(torch.mean(s**2))
        return super().right_inverse(weight / rms_scale), rms_scale


def scaled_orthogonal(module: nn.Module, name: str = "weight"):
    """Register a scaled orthogonal parametrization for a weight matrix in a module."""
    orth = ScaledOrthogonalParametrization()
    register_parametrization(module, name, orth, unsafe=True)
    return module


__all__ = [
    "low_rank",
    "orthogonal",
    "scaled_orthogonal",
]
