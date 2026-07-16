from typing import NamedTuple, Optional, Literal

import torch

from nn_lib.utils import RunningCovariance, inv_sqrt_spd, orthogonalize
from nn_lib.utils.linalg import _truncate_svd


class XValStats(NamedTuple):
    """Fully-specified 'global' statistics for the cross-validated nuclear-norm estimators (
    `xval_nuc_norm_cross_cov` and friends). These estimators require two passes through the data.
    This object contains the relevant summary statistics after the first pass. The preferred way
    to instantiate XValStats objects is with `precompute_xval_stats`, which computes the SVD and
    enforces the centered/uncentered invariants.

    Callers always pass *raw* (uncentered) batches to the estimators; the XValStats object then
    provides information about whether and how to do the centering.

    Fields:
        u, s, vh: SVD of the full-data cross-covariance ``matX.T @ matY / m_total``. This cross-cov
            must be *centered* iff means are provided below. full_matrices=False is preferred, but
            full_matrices=True is tolerated (extra columns are sliced off internally).
        m_total: total number of samples the SVD (and means) were computed from.
        dof: degrees of freedom, i.e. 1 if centering with empirical means and 0 otherwise
        mean_x, mean_y: full-data means of X and Y. Leaving mean_x and mean_y as None indicates
            to the various xval_nuc_norm_cross_cov functions that mu=0 and dof=0 and data are not
            to be centered.
    """

    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    m_total: int
    dof: int
    mean_x: torch.Tensor
    mean_y: torch.Tensor

    @staticmethod
    def from_running_covariance(cov_xy: RunningCovariance) -> "XValStats":
        u, s, vh = _truncate_svd(*torch.linalg.svd(cov_xy.covariance, full_matrices=False))
        return XValStats(
            u,
            s,
            vh,
            cov_xy.count,
            dof=1 if cov_xy.centered else 0,
            mean_x=cov_xy.mu_x,
            mean_y=cov_xy.mu_y,
        )


def _xval_stats_from_full_data(matX: torch.Tensor, matY: torch.Tensor, centered: bool) -> XValStats:
    """Build `XValStats` treating matX/matY as the entire dataset (the convenience path used when
    no precomputed `stats` are supplied)."""
    rc = RunningCovariance(centered=centered)
    rc.update(matX, matY)
    return XValStats.from_running_covariance(rc)


@torch.jit.script
def _prepare_xval(
    matX: torch.Tensor,
    matY: torch.Tensor,
    stats: XValStats,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Shared preprocessing for the cross-validated nuclear norm estimators.

    Centering is decided entirely by `stats`: if it carries means, each incoming batch is centered
    with them here (so callers always pass raw batches) and the (m_total - 1) covariance
    normalization is used; otherwise the data is used as-is with mu = 0 assumed known.

    Centered leave-one-out math: with full-data means mu_x, mu_y over M = m_total samples and
    centered samples x_c = x_i - mu_x, y_c = y_i - mu_y, the leave-one-out centered scatter
    satisfies

        S^{-i} := sum_{j != i} (x_j - mu_x^{-i})(y_j - mu_y^{-i})^T = S - M/(M-1) * x_c y_c^T,

    where S = sum_j (x_j - mu_x)(y_j - mu_y)^T is the full centered scatter. The polar factor is
    invariant to positive scaling, so we may work with any positive multiple of S^{-i}. Dividing
    by the full-data covariance normalization (M-1) gives, with cov_xy = S/(M-1),

        polar(S^{-i}) = polar( cov_xy - (M/(M-1)^2) * x_c y_c^T ).

    Both the centered and uncentered cases are captured by a single downdate denominator

        cov_xy_denom = (M - dof)^2 / M          # centered (dof=1): (M-1)^2/M -> coeff M/(M-1)^2
                                                # uncentered (dof=0): M       -> coeff 1/M

    Additionally, for the evaluation vectors,

        x_i - mu_x^{-i} = M/(M-1) * (x_i - mu_x),

    so evaluating the bilinear form with downdated-mean-centered vectors is the same as using
    full-mean-centered vectors scaled by (M/(M-1))^2 = M^2/(M-dof)^2 (which is 1 when uncentered).

    Returns (matX, matY, cov_xy_denom, downdate_mean_factor) such that each estimator downdates
    by ``x y^T / cov_xy_denom`` and multiplies the final per-sample values by
    ``downdate_mean_factor``.
    """
    # No special logic for centering is required in this function since XValStats is expected to
    # have means set to zero and dof set to 0 in the uncentered case.
    matX = matX - stats.mean_x.unsqueeze(0)
    matY = matY - stats.mean_y.unsqueeze(0)
    # Downdate denominator (M-dof)^2/M: reproduces the recompute-from-scratch LOO estimand exactly
    # (uncentered -> M, i.e. unchanged; centered -> (M-1)^2/M, i.e. coefficient M/(M-1)^2).
    cov_xy_denom = float(stats.m_total - stats.dof) ** 2 / float(stats.m_total)
    # mean downdates apply a 1.0 multiplier if uncentered or a m/(m-1) multiplier twice,
    # once for x and once for y, if centered.
    downdate_mean_factor = float(stats.m_total) ** 2 / float(stats.m_total - stats.dof) ** 2

    return matX, matY, cov_xy_denom, downdate_mean_factor


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

    Key identity: we never need the downdated singular vectors themselves, only the bilinear form
    x_i^T polar(M_i) y_i where M_i = u diag(s) vh - x_i y_i^T / denom and polar(.) = U V^T from
    the SVD. Writing M_i in the augmented orthonormal bases U_bar = [u, x_perp/alpha], V_bar = [
    vh.T, y_perp/beta], we have x_i = U_bar @ [p_i; alpha_i] exactly (and likewise y_i), so

        x_i^T polar(M_i) y_i = [p_i; alpha_i]^T polar(K_i) [q_i; beta_i]

    where K_i = diag([s, 0]) - [p_i; alpha_i] [q_i; beta_i]^T / denom is the small (r+1 x r+1)
    core matrix. This removes any dependence on the ambient dimensions beyond the initial
    projections, and everything is batched over samples.

    If alpha_i (or beta_i) is exactly/numerically zero, the corresponding row (or column) of K_i
    is ~zero and contributes only O(alpha_i) to the bilinear form, so no special-casing or
    tolerance branching is required.

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
    method: Literal["brute_force", "rank1", "ab", "orthogonalize", "secular"] = "brute_force",
    k: Optional[int] = None,
    centered: Optional[bool] = None,
    stats: Optional[XValStats] = None,
) -> torch.Tensor:
    """Calculate the cross-validated (leave-one-out) nuclear norm of the cross-covariance matrix
    between matX and matY.

    Callers always pass raw (uncentered) matX/matY here. Centering behavior is controlled by passing
    either 'centered' as a boolean flag (if matX and matY are the full data) or by passing 'stats'
    (if matX and matY are batches and stats were precomputed).

    :param matX: batch of samples from distribution X.
    :param matY: batch of samples from distribution Y.
    :param method: "brute_force", "rank1", "ab", "orthogonalize", "secular", controls dispatch to
        various accelerated estimators. brute_force is slow but exact.
    :param k: optionally truncate the global SVD to rank k before downdating.
    :param centered: whether to subtract empirical means from x and y when estimating cov_xy. Only
        use this argument if matX and matY are the full data. If calculating batch-wise, pass in
        'stats' instead.
    :param stats: precomputed full-data statistics. Recommended construction of 'stats' is using
        XValStats.from_running_covariance()
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

    if stats is None:
        if centered is None:
            raise ValueError("Specify either 'stats' (if batching) or 'centered' (if full data).")
        stats = _xval_stats_from_full_data(matX, matY, centered)
    elif centered is not None:
        raise ValueError("Specify only one of 'stats' (if batching) or 'centered' (if full data).")

    if method == "brute_force":
        if k is not None:
            raise ValueError("Low-rank k argument is not supported in brute-force method")
        return xval_nuc_norm_cross_cov_brute_force(matX, matY, stats)
    elif method == "rank1":
        return xval_nuc_norm_cross_cov_rank1(matX, matY, stats, k=k)
    elif method == "ab":
        return xval_nuc_norm_cross_cov_ab(matX, matY, stats, k=k)
    elif method == "orthogonalize":
        return xval_nuc_norm_cross_cov_orthogonalize(matX, matY, stats, k=k)
    elif method == "secular":
        return xval_nuc_norm_cross_cov_secular(matX, matY, stats, k=k)
    else:
        raise ValueError(f"method {method} is not supported")


@torch.jit.script
def xval_nuc_norm_cross_cov_brute_force(
    matX: torch.Tensor, matY: torch.Tensor, stats: XValStats
) -> torch.Tensor:
    """Reference implementation: explicitly forms and decomposes each downdated
    cross-covariance matrix.

    :param matX: input matrix of shape (m, n_x) where m is batch size
    :param matY: input matrix of shape (m, n_y)
    :param stats: precomputed full-data statistics (see `XValStats`)
    """
    matX, matY, denom, scale = _prepare_xval(matX, matY, stats)
    u, s, vh = stats.u, stats.s, stats.vh
    # Restore the full cross-covariance matrix
    cross_cov = (u * s) @ vh
    m = matX.shape[0]
    vals = []
    for i in range(m):
        x, y = matX[i, :], matY[i, :]
        # Downdate the cross-covariance
        xcov_i = cross_cov - x[:, None] * y[None, :] / denom
        u_i, _, vh_i = torch.linalg.svd(xcov_i, full_matrices=False)
        vals.append(y @ (vh_i.T @ (u_i.T @ x)))
    # Downdating the means is conveniently equivalent to scaling the final result
    return scale * torch.stack(vals).mean()


@torch.jit.script
def xval_nuc_norm_cross_cov_rank1(
    matX: torch.Tensor,
    matY: torch.Tensor,
    stats: XValStats,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Leave-one-out estimator via rank-1 downdates of the global SVD, fully batched.

    Uses the exact polar factor of the small augmented core matrices (see
    `_augmented_core`), computed with a single batched SVD. Exact up to floating point
    (matches `brute_force` when k is None).
    """
    matX, matY, denom, scale = _prepare_xval(matX, matY, stats)
    u, s, vh = _truncate_svd(stats.u, stats.s, stats.vh, k)
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
def xval_nuc_norm_cross_cov_ab(
    matX: torch.Tensor,
    matY: torch.Tensor,
    stats: XValStats,
    k: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Leave-one-out estimator computing the polar factor of the small augmented core
    matrices K (see `_augmented_core`) as (K K^T)^{-1/2} K, with a single batched eigh
    replacing the per-sample loop. Exact up to floating point (and the eps clamp on
    eigenvalues; note the Gram matrix K K^T squares the condition number, so prefer
    the `rank1` method in float32).
    """
    matX, matY, denom, scale = _prepare_xval(matX, matY, stats)
    u, s, vh = _truncate_svd(stats.u, stats.s, stats.vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    # Polar factor of each K via (K K^T)^{-1/2} K, batched
    matC = inv_sqrt_spd(K @ K.transpose(-2, -1), eps=eps)

    # value_i = Qt_i^T K_i^T C_i Pt_i
    vals = torch.einsum("bj,bij,bil,bl->b", Qt, K, matC, Pt)
    return scale * vals.mean()


@torch.jit.script
def xval_nuc_norm_cross_cov_orthogonalize(
    matX: torch.Tensor,
    matY: torch.Tensor,
    stats: XValStats,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Leave-one-out estimator using batched Newton-Schulz orthogonalization of the small
    augmented core matrices (approximate polar factor; matmul-only, so it batches and
    parallelizes especially well on GPU). Approximate: singular values are mapped into a
    band around 1 rather than exactly to 1, so expect small relative error vs brute_force.
    """
    matX, matY, denom, scale = _prepare_xval(matX, matY, stats)
    u, s, vh = _truncate_svd(stats.u, stats.s, stats.vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    vals = torch.einsum("bi,bij,bj->b", Pt, orthogonalize(K), Qt)
    return scale * vals.mean()


@torch.jit.script
def xval_nuc_norm_cross_cov_secular(
    matX: torch.Tensor,
    matY: torch.Tensor,
    stats: XValStats,
    k: Optional[int] = None,
    eps: float = 1e-18,
) -> torch.Tensor:
    """Leave-one-out estimator via the Gandhi-Rajgor (2017) secular / Cauchy structure,
    specialized to the bilinear form (never forms a singular vector).

    Background: the other fast methods obtain polar(K_i) densely (SVD / (KK^T)^{-1/2}K /
    Newton-Schulz). Gandhi & Rajgor instead update a rank-1-perturbed SVD by (a) solving a
    secular equation for the updated singular values mu_j and (b) assembling the updated singular
    vectors as columns of a Cauchy matrix (FMM-accelerated). We only ever want the scalar pbar^T
    polar(K_i) qbar, so we can skip the singular vectors entirely.
    """
    matX, matY, denom, scale = _prepare_xval(matX, matY, stats)
    u, s, vh = _truncate_svd(stats.u, stats.s, stats.vh, k)
    Pt, Qt, K = _augmented_core(matX, matY, u, s, vh, denom)

    d = K.shape[1]
    rho = 1.0 / denom
    sigma = torch.cat([s, torch.zeros(1, device=s.device, dtype=s.dtype)])  # (d,)
    max_rank = min(matX.shape[1], matY.shape[1])

    # Updated singular values mu_j (values only, batched): (b, d)
    mu = torch.linalg.svdvals(K)
    mu2 = mu * mu

    # Pole denominators sigma_k^2 - mu_j^2: (b, j, k)
    s2 = (sigma * sigma).view(1, 1, d)
    den = s2 - mu2.unsqueeze(2)
    # Removable singularities (mu_j^2 == sigma_k^2) are measure-zero; floor magnitude to avoid inf.
    den = torch.where(den.abs() < eps, torch.full_like(den, eps), den)

    P = Pt.unsqueeze(1)  # (b, 1, k)
    Q = Qt.unsqueeze(1)  # (b, 1, k)
    sig = sigma.view(1, 1, d)

    Phi_pp = torch.sum(P * P / den, dim=2)  # (b, j)
    Phi_pq = torch.sum(sig * P * Q / den, dim=2)

    # 2x2 null vector of [[a, rho*mu*Phi_pp],[rho*mu*Phi_qq, a]] with a = rho*Phi_pq - 1.
    # At a true root a^2 = rho^2 mu^2 Phi_pp Phi_qq, so (t, s) ~ (rho*mu*Phi_pp, -a) is null;
    # its internal sign is fixed (only the overall sign is free, and t*s is invariant to it).
    a = rho * Phi_pq - 1.0
    t_un = rho * mu * Phi_pp  # (b, j)
    s_un = -a  # (b, j)

    # Cauchy singular vectors (unnormalized) from (t_un, s_un): (b, j, k)
    mu_col = mu.unsqueeze(2)
    tj = t_un.unsqueeze(2)
    sj = s_un.unsqueeze(2)
    uhat = rho * (tj * sig * Q + mu_col * sj * P) / den
    vhat = rho * (sj * sig * P + mu_col * tj * Q) / den

    nu = torch.linalg.norm(uhat, dim=2)  # (b, j)
    nv = torch.linalg.norm(vhat, dim=2)
    nu = torch.where(nu < eps, torch.full_like(nu, eps), nu)
    nv = torch.where(nv < eps, torch.full_like(nv, eps), nv)

    # Normalized scalar projections t_j = pbar^T uhat_j, s_j = qbar^T vhat_j
    t = torch.sum(P * uhat, dim=2) / nu  # (b, j)
    sproj = torch.sum(Q * vhat, dim=2) / nv

    ts = t * sproj  # (b, j)
    # Drop components beyond the LOO matrix's max possible rank (their mu_j ~ 0 and the
    # corresponding t_j s_j ~ 0 already; masking keeps parity with the other methods' truncation).
    if d > max_rank:
        ts = ts[:, :max_rank]
    vals = ts.sum(dim=1)  # (b,)
    return scale * vals.mean()


__all__ = [
    "XValStats",
    "xval_nuc_norm_cross_cov",
]
