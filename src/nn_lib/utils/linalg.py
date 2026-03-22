import torch


@torch.jit.script
def rank_one_svd_update(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given matrix M and its singular value decomposition such that M = U @ S @ Vh, efficiently
    compute the SVD of (M + x @ y.T)
    """
    # Ensure S is a 1-D vector of singular values
    if S.ndim == 2 and S.shape[0] == S.shape[1]:
        S = torch.diag(S)

    # Ensure x, y are 1-D vectors on same device/dtype
    x = x.flatten()
    y = y.flatten()

    # get rows (m) rank (r) and cols (n) of original matrix
    m, r, n = U.shape[0], S.shape[0], Vh.shape[1]

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
    max_sigma = S.abs().max() if r > 0 else torch.zeros_like(S[0])
    tol = max(m, n) * float(eps) * float(max(norm_x, norm_y, max_sigma, 1.0))

    alpha_nonzero = alpha.item() > tol
    beta_nonzero = beta.item() > tol

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


__all__ = [
    "rank_one_svd_update",
]
