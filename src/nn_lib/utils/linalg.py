from typing import Optional, Literal, NamedTuple

import torch


def eye_like(a: torch.Tensor) -> torch.Tensor:
    return torch.eye(a.shape[0], device=a.device, dtype=a.dtype)


class XValStats(NamedTuple):
    """Container for precomputed full-data statistics used by the cross-validated
    nuclear norm estimators (`xval_nuc_norm_cross_cov` and friends).

    Fields:
        u, s, vh: SVD of the full-data cross-covariance matX.T @ matY / m_total. The cross-cov
            may or may not be centered depending on the context (i.e. if means are not None).
            full_matrices=False is preferred, but full_matrices=True is tolerated (extra columns are
            sliced off internally).
        m_total: total number of samples the SVD (and means) were computed from.
        mean_x, mean_y: full-data means of X and Y; required when calling the
            estimators with center=True, may be None otherwise.
    """

    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    m_total: int
    mean_x: Optional[torch.Tensor] = None
    mean_y: Optional[torch.Tensor] = None


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
def _prepare_xval(
    matX: torch.Tensor,
    matY: torch.Tensor,
    center: bool,
    stats: Optional[XValStats],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Shared preprocessing for the cross-validated nuclear norm estimators.

    Handles (a) obtaining the global SVD (from `stats` or by computing it from the
    given full data), and (b) the mean-downdating math when center=True.

    Centered leave-one-out math: with full-data means mu_x, mu_y over M = m_total
    samples and centered samples x_c = x_i - mu_x, y_c = y_i - mu_y, the leave-one-out
    centered scatter satisfies

        sum_{j != i} (x_j - mu_x^{-i})(y_j - mu_y^{-i})^T = S - M/(M-1) * x_c y_c^T,

    i.e. the downdated *centered* cross-covariance is (up to positive scale, which is
    irrelevant to the polar factor) the same rank-1 downdate as before with coefficient
    1/(M-1) instead of 1/M and with fully-centered vectors. Additionally,

        x_i - mu_x^{-i} = M/(M-1) * (x_i - mu_x),

    so evaluating the bilinear form with downdated-mean-centered vectors is the same as
    using full-mean-centered vectors scaled by (M/(M-1))^2.

    Returns (matX_centered, matY_centered, u, s, vh, downdat_cov_denom, downdate_mean_factor)
    such that each estimator downdates by x y^T / downdat_cov_denom and multiplies the
    final per-sample values by downdate_mean_factor.
    """
    if stats is None:
        m_total = matX.shape[0]
        if center:
            matX = matX - matX.mean(dim=0)
            matY = matY - matY.mean(dim=0)
        cross_cov = matX.T @ matY / m_total
        u, s, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    else:
        u, s, vh = stats.u, stats.s, stats.vh
        m_total = stats.m_total
        if center:
            mean_x = stats.mean_x
            mean_y = stats.mean_y
            if mean_x is None:
                raise ValueError("center=True requires stats.mean_x to be provided")
            if mean_y is None:
                raise ValueError("center=True requires stats.mean_y to be provided")
            matX = matX - mean_x.unsqueeze(0)
            matY = matY - mean_y.unsqueeze(0)

    if center:
        if m_total < 2:
            raise ValueError("center=True requires m_total >= 2")
        downdat_cov_denom = float(m_total - 1)
        downdate_mean_factor = float(m_total) ** 2 / float(m_total - 1) ** 2
    else:
        downdat_cov_denom = float(m_total)
        downdate_mean_factor = 1.0

    return matX, matY, u, s, vh, downdat_cov_denom, downdate_mean_factor


@torch.jit.script
def _augmented_core(
    matX: torch.Tensor,
    matY: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
    downdate_denom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared construction for the fast leave-one-out estimators.

    Key identity: we never need the downdated singular vectors themselves, only the
    bilinear form x_i^T polar(M_i) y_i where M_i = u diag(s) vh - x_i y_i^T / denom
    and polar(.) = U V^T from the SVD. Writing M_i in the augmented orthonormal bases
    U_bar = [u, x_perp/alpha], V_bar = [vh.T, y_perp/beta], we have
    x_i = U_bar @ [p_i; alpha_i] exactly (and likewise y_i), so

        x_i^T polar(M_i) y_i = [p_i; alpha_i]^T polar(K_i) [q_i; beta_i]

    where K_i = diag([s, 0]) - [p_i; alpha_i] [q_i; beta_i]^T / denom is the small
    (r+1 x r+1) core matrix. This removes any dependence on the ambient dimensions
    beyond the initial projections, and everything is batched over samples.

    If alpha_i (or beta_i) is exactly/numerically zero, the corresponding row (or
    column) of K_i is ~zero and contributes only O(alpha_i) to the bilinear form, so
    no special-casing or tolerance branching is required.

    Returns (Pt, Qt, K) with shapes (m, r+1), (m, r+1), (m, r+1, r+1).
    """
    # Project all samples into the singular subspaces at once: (m, r)
    P = matX @ u
    Q = matY @ vh.T

    # Norms of the residuals orthogonal to the subspaces: (m,)
    alpha = torch.linalg.norm(matX - P @ u.T, dim=1)
    beta = torch.linalg.norm(matY - Q @ vh, dim=1)

    # Augmented coordinates [p; alpha], [q; beta]: (m, r+1)
    Pt = torch.cat([P, alpha.unsqueeze(1)], dim=1)
    Qt = torch.cat([Q, beta.unsqueeze(1)], dim=1)

    # Batched core matrices K_i = diag([s, 0]) - Pt_i Qt_i^T / denom: (m, r+1, r+1)
    s_aug = torch.cat([s, torch.zeros(1, device=s.device, dtype=s.dtype)])
    K = torch.diag_embed(s_aug).unsqueeze(0) - Pt.unsqueeze(2) * Qt.unsqueeze(1) / downdate_denom
    return Pt, Qt, K


def xval_nuc_norm_cross_cov(
    matX: torch.Tensor,
    matY: torch.Tensor,
    method: Literal["brute_force", "rank1", "ab", "orthogonalize"] = "brute_force",
    center: bool = False,
    k: Optional[int] = None,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Calculate the cross-validated (leave-one-out) nuclear norm of the cross-covariance
    matrix between matX and matY.

    :param center: if True, means are treated as estimated from data: sample i's
        contribution is removed from the means as well as from the cross-covariance,
        i.e. each term is (y_i - mu_y^{-i})^T V^{-i} (U^{-i})^T (x_i - mu_x^{-i}) where
        the ^{-i} quantities are computed with sample i held out. If False, mu = 0 is
        assumed known and data is used as-is.
    :param k: optionally truncate the global SVD to rank k before downdating.
    :param stats: precomputed full-data statistics (see `XValStats` and
        `precompute_xval_stats`). Required when calling batch-by-batch so that every
        batch downdates the same 'global' SVD/means: one pass to precompute stats, a
        second pass to call this per batch. When None, matX/matY are treated as the
        full dataset and stats are computed internally. When center=True, stats must
        include mean_x and mean_y (computed over the same m_total samples as the SVD,
        which must be of the *centered* cross-covariance).
    """
    if matX.size(0) != matY.size(0):
        raise ValueError(
            f"The number of rows of matX and matY should be the same "
            f"but got {matX.shape} and {matY.shape}"
        )
    if matX.ndim != 2:
        raise ValueError(f"X must be 2-dimensional")
    if matY.ndim != 2:
        raise ValueError(f"Y must be 2-dimensional")
    if center and stats is not None and (stats.mean_x is None or stats.mean_y is None):
        raise ValueError("center=True requires stats.mean_x and stats.mean_y to be provided")

    if method == "brute_force":
        if k is not None:
            raise ValueError("Low-rank k argument is not supported in brute-force method")
        return xval_nuc_norm_cross_cov_brute_force(matX, matY, center=center, stats=stats)
    elif method == "rank1":
        return xval_nuc_norm_cross_cov_rank1(matX, matY, center=center, k=k, stats=stats)
    elif method == "ab":
        return xval_nuc_norm_cross_cov_ab(matX, matY, center=center, k=k, stats=stats)
    elif method == "orthogonalize":
        return xval_nuc_norm_cross_cov_orthogonalize(matX, matY, center=center, k=k, stats=stats)
    else:
        raise ValueError(f"method {method} is not supported")


@torch.jit.script
def xval_nuc_norm_cross_cov_brute_force(
    matX: torch.Tensor,
    matY: torch.Tensor,
    center: bool = False,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Reference implementation: explicitly forms and decomposes each downdated
    cross-covariance matrix.

    :param matX: input matrix of shape (m, n_x) where m is batch size
    :param matY: input matrix of shape (m, n_y)
    :param center: whether the cross-covariance is centered using means estimated from the data and
        therefore requires 1/(m_total-1) adjustments on covariances.
    :param stats: precomputed statistics from all m_total full data. If omitted, the current m
        samples are treated as the full data.
    """
    matX, matY, u, s, vh, denom, scale = _prepare_xval(matX, matY, center, stats)
    u, s, vh = _truncate_svd(u, s, vh)
    # Restore the full cross-covariance matrix
    cross_cov = (u * s) @ vh
    m = matX.shape[0]
    vals = []
    for i in range(m):
        x, y = matX[i, :], matY[i, :]
        # Note that x and y are already centered using full-data means  by _prepare_xval if
        # center=True. This effectively subtracts (x-mu_x)*(y-mu_y)/(m-1). Scaling xcov_i by (
        # m-1)/(m-2) would then give us the xcov_without_i cross covariance, but that scaling
        # doesn't matter for the purposes of getting u_i and vh_i. In the case where
        # center=False, denom=m_total and this again is the correct subtraction but unscaled xcov_i.
        xcov_i = cross_cov - x[:, None] * y[None, :] / denom
        u_i, _, vh_i = torch.linalg.svd(xcov_i, full_matrices=False)
        vals.append(y @ (vh_i.T @ (u_i.T @ x)))
    return scale * torch.stack(vals).mean()


@torch.jit.script
def xval_nuc_norm_cross_cov_rank1(
    matX: torch.Tensor,
    matY: torch.Tensor,
    center: bool = False,
    k: Optional[int] = None,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Leave-one-out estimator via rank-1 downdates of the global SVD, fully batched.

    Uses the exact polar factor of the small augmented core matrices (see
    `_augmented_core`), computed with a single batched SVD. Exact up to floating point
    (matches `brute_force` when k is None).
    """
    matX, matY, u, s, vh, denom, scale = _prepare_xval(matX, matY, center, stats)
    u, s, vh = _truncate_svd(u, s, vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    U_k, _, Vh_k = torch.linalg.svd(K, full_matrices=False)

    # Drop excess components if the augmented rank exceeds the max possible rank of the
    # downdated matrix (their singular values are exactly zero and their singular vectors
    # arbitrary; matches the truncation previously done in rank_one_svd_update).
    max_rank = min(matX.shape[1], matY.shape[1])
    if K.shape[1] > max_rank:
        U_k = U_k[:, :, :max_rank]
        Vh_k = Vh_k[:, :max_rank, :]

    vals = torch.einsum("bi,bik,bkj,bj->b", Pt, U_k, Vh_k, Qt)
    return scale * vals.mean()


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
def xval_nuc_norm_cross_cov_ab(
    matX: torch.Tensor,
    matY: torch.Tensor,
    center: bool = False,
    k: Optional[int] = None,
    eps: float = 1e-12,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Leave-one-out estimator computing the polar factor of the small augmented core
    matrices K (see `_augmented_core`) as (K K^T)^{-1/2} K, with a single batched eigh
    replacing the per-sample loop. Exact up to floating point (and the eps clamp on
    eigenvalues; note the Gram matrix K K^T squares the condition number, so prefer
    the `rank1` method in float32).
    """
    matX, matY, u, s, vh, denom, scale = _prepare_xval(matX, matY, center, stats)
    u, s, vh = _truncate_svd(u, s, vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    # Polar factor of each K via (K K^T)^{-1/2} K, batched
    matC = inv_sqrt_spd(K @ K.transpose(-2, -1), eps=eps)

    # value_i = Qt_i^T K_i^T C_i Pt_i
    vals = torch.einsum("bj,bij,bil,bl->b", Qt, K, matC, Pt)
    return scale * vals.mean()


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


@torch.jit.script
def xval_nuc_norm_cross_cov_orthogonalize(
    matX: torch.Tensor,
    matY: torch.Tensor,
    center: bool = False,
    k: Optional[int] = None,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Leave-one-out estimator using batched Newton-Schulz orthogonalization of the small
    augmented core matrices (approximate polar factor; matmul-only, so it batches and
    parallelizes especially well on GPU). Approximate: singular values are mapped into a
    band around 1 rather than exactly to 1, so expect small relative error vs brute_force.
    """
    matX, matY, u, s, vh, denom, scale = _prepare_xval(matX, matY, center, stats)
    u, s, vh = _truncate_svd(u, s, vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    vals = torch.einsum("bi,bij,bj->b", Pt, orthogonalize(K), Qt)
    return scale * vals.mean()


__all__ = [
    "XValStats",
    "eye_like",
    "inv_sqrt_spd",
    "orthogonalize",
    "rank_one_svd_update",
    "xval_nuc_norm_cross_cov",
]
