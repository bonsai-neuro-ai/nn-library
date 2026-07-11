from typing import Optional

import torch


def eye_like(a: torch.Tensor) -> torch.Tensor:
    return torch.eye(a.shape[0], device=a.device, dtype=a.dtype)


@torch.jit.script
def _truncate_svd(
    u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, k: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice an SVD triple down to rank r = len(s) (handles full_matrices=True inputs,
    where u and vh carry orthogonal-complement columns/rows paired with no singular value),
    then optionally truncate further to rank k.
    """
    r = s.shape[0]
    if u.shape[1] > r:
        u = u[:, :r]
    if vh.shape[0] > r:
        vh = vh[:r, :]
    if k is not None and k < r:
        u = u[:, :k]
        vh = vh[:k, :]
        s = s[:k]
    return u, s, vh


@torch.jit.script
def rank_one_svd_update(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given matrix the singular value decomposition of some matrix M such that M = U @ S @ Vh,
    efficiently compute the updated SVD of (M + x @ y.T), i.e. a rank one perturbation to M
    """
    # Ensure S is a 1-D vector of singular values
    if S.ndim == 2 and S.shape[0] == S.shape[1]:
        S = torch.diag(S)

    # Ensure x, y are 1-D vectors on same device/dtype
    x = x.flatten()
    y = y.flatten()

    # get rows (m) rank (r) and cols (n) of original matrix
    m, r, n = U.shape[0], S.shape[0], Vh.shape[1]

    # Fix shape issues (enforce consistent (n, r), (r,), (r, m) shapes where r=rank)
    U, S, Vh = _truncate_svd(U, S, Vh)

    # Project x,y into the current subspaces
    # p in R^r, q in R^r
    if r > 0:
        p = U.T @ x
        q = Vh @ y
    else:
        p = torch.zeros_like(S)
        q = torch.zeros_like(S)

    # Compute residuals (components orthogonal to U and V)
    x_perp = x - (U @ p if r > 0 else torch.zeros_like(x))
    y_perp = y - (Vh.T @ q if r > 0 else torch.zeros_like(y))

    alpha = torch.linalg.norm(x_perp)
    beta = torch.linalg.norm(y_perp)

    # Tolerance to consider a residual as numerically zero
    norm_x = torch.linalg.norm(x)
    norm_y = torch.linalg.norm(y)
    max_sigma = S.abs().max() if r > 0 else torch.zeros_like(alpha)
    tol = max(m, n) * float(eps) * float(max(norm_x, norm_y, max_sigma, 1.0))

    alpha_nonzero = alpha > tol
    beta_nonzero = beta > tol

    # Build augmented bases U_bar (m x ru) and V_bar (n x rv)
    U_bar_cols = [U]
    if alpha_nonzero:
        u_perp = x_perp / alpha
        U_bar_cols.append(u_perp.unsqueeze(1))
    U_bar = torch.cat(U_bar_cols, dim=1) if len(U_bar_cols) > 1 else U

    V_bar_cols = [Vh.T]
    if beta_nonzero:
        v_perp = y_perp / beta
        V_bar_cols.append(v_perp.unsqueeze(1))
    V_bar = torch.cat(V_bar_cols, dim=1) if len(V_bar_cols) > 1 else Vh.T

    ru = r + (1 if alpha_nonzero else 0)
    rv = r + (1 if beta_nonzero else 0)

    # Build the small correction matrix K of shape (ru, rv): K = Sigma_bar + P @ Q^T
    Sigma_bar = torch.zeros((ru, rv), device=S.device, dtype=S.dtype)
    if r > 0:
        Sigma_bar[:r, :r] = torch.diag(S)

    P = torch.cat([p, alpha.unsqueeze(0)]) if alpha_nonzero else p
    Q = torch.cat([q, beta.unsqueeze(0)]) if beta_nonzero else q

    # Ensure P and Q are 1-D of correct length
    P = P.reshape(ru)
    Q = Q.reshape(rv)

    K = Sigma_bar + P.unsqueeze(1) @ Q.unsqueeze(0)

    # SVD of the small matrix K (rectangular allowed)
    # K = U_k @ S_k @ Vh_k where U_k: (ru x k), Vh_k: (k x rv) and k = min(ru, rv)
    U_k, S_k, Vh_k = torch.linalg.svd(K, full_matrices=False)

    # Form updated full U and V
    U_new = U_bar @ U_k
    V_new = V_bar @ Vh_k.T

    # Drop any excess singular values/vectors if rank exceeds maximum possible rank
    max_rank = min(m, n)
    if ru > max_rank or rv > max_rank:
        U_new = U_new[:, :max_rank]
        S_k = S_k[:max_rank]
        V_new = V_new[:, :max_rank]

    # Return new SVD: U_new @ diag(S_k) @ V_new.T
    return U_new, S_k, V_new.T


@torch.jit.script
def inv_sqrt_spd(B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute B^{-1/2} for SPD B using eigendecomposition. Supports batched input
    of shape (..., n, n).

    Note: eigenvalues are clamped at eps before the inverse square root; for inputs
    that are Gram matrices M @ M.T this means singular values of M are effectively
    floored at sqrt(eps).
    """
    evals, evecs = torch.linalg.eigh(B)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt = (evecs * evals.rsqrt().unsqueeze(-2)) @ evecs.transpose(-2, -1)
    return inv_sqrt


@torch.jit.script
def orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """Approximate orthogonalization of a matrix using a fixed number of Newton-Schulz iterations
    with carefully chosen coefficients for stability. Supports batched input of shape
    (..., m, n); each matrix in the batch is normalized and orthogonalized independently.

    This code is adapted from github.com/modula/modula with the following license:

        Copyright (c) 2024 Jeremy Bernstein

        Permission is hereby granted, free of charge, to any person obtaining a copy of this
        software and associated documentation files (the "Software"), to deal in the Software
        without restriction, including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software, and to permit
        persons to whom the Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all copies or
        substantial portions of the Software.
    """

    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]

    transpose = M.shape[-1] > M.shape[-2]
    if transpose:
        M = M.transpose(-2, -1)
    # Per-matrix Frobenius normalization (batched)
    M = M / torch.sqrt((M * M).sum(dim=[-2, -1], keepdim=True))
    for a, b, c in abc_list:
        A = M.transpose(-2, -1) @ M
        I = torch.eye(A.shape[-1], device=M.device, dtype=M.dtype)
        M = M @ (a * I + b * A + c * (A @ A))
    if transpose:
        M = M.transpose(-2, -1)
    return M


__all__ = [
    "eye_like",
    "inv_sqrt_spd",
    "orthogonalize",
    "rank_one_svd_update",
]
