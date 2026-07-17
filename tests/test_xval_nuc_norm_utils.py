import unittest

import numpy as np
import torch
from torch.testing import assert_close as assert_close_torch

from nn_lib.utils import xval_nuc_norm_cross_cov, orthogonalize, RunningCovariance, XValStats

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def assert_close(x, y, lenience=0.0, atol=None, rtol=None):
    """Compare two tensors.

    By default the tolerance is tied to the dtype's machine epsilon (via `lenience`), which is
    the right guard for the *exact* estimators. For the *approximate* `orthogonalize` method,
    pass explicit `atol`/`rtol`: its error is a fixed Newton-Schulz approximation band
    (~3e-3 absolute on the bilinear form; see code/measure_ortho_band.py and notes/methods.md
    §1), independent of dtype, so an eps-scaled tolerance is inappropriate (it is far too tight
    at float64 and only coincidentally passes at float32).
    """
    if atol is None:
        atol = (10.0**lenience) * np.sqrt(torch.finfo(x.dtype).eps)
    if rtol is None:
        rtol = atol / 10
    assert_close_torch(x, y, rtol=rtol, atol=atol)


# Fixed, dtype-independent tolerance for the approximate `orthogonalize` method. The intrinsic
# Newton-Schulz error on the averaged estimator is ~3e-3 absolute on test-scale inputs; 1e-2
# leaves ~3x headroom while still catching any gross regression. Relative comparisons of two
# orthogonalize outputs (e.g. X<->Y symmetry) stay exact and keep the tight default tolerance.
ORTHO_ATOL, ORTHO_RTOL = 1e-2, 2e-2


def _exact_polar(M: torch.Tensor) -> torch.Tensor:
    """Exact orthogonal polar factor U V^T (batched) -- the quantity that the fast Newton-Schulz
    `orthogonalize` kernel approximates. Used to test the kernel's accuracy in isolation from the
    nuclear-norm estimator logic."""
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def _recompute_loo_reference(x: torch.Tensor, y: torch.Tensor, centered: bool) -> torch.Tensor:
    """Independent, from-scratch leave-one-out nuclear-norm estimand (no downdate tricks).

    For each held-out sample i, drop it, recompute the LOO mean and cross-covariance over the
    remaining M-1 samples, take the polar factor, and evaluate at the LOO-centered held-out
    vectors. This is the estimand defined in notes/methods.md §0; it deliberately shares no code
    with `_prepare_xval`, so it catches errors in the centering / downdate-coefficient algebra
    (which method-vs-brute_force tests cannot, since all methods share `_prepare_xval`).
    """
    m = x.shape[0]
    vals = []
    for i in range(m):
        mask = torch.ones(m, dtype=torch.bool, device=x.device)
        mask[i] = False
        x_drop_i, y_drop_i = x[mask], y[mask]
        if centered:
            mux_drop_i, muy_drop_i = x_drop_i.mean(0), y_drop_i.mean(0)
        else:
            mux_drop_i = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
            muy_drop_i = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
        cov_i = (x_drop_i - mux_drop_i).T @ (
            y_drop_i - muy_drop_i
        )  # positive scale irrelevant to polar factor
        u_drop_i, _, vh_drop_i = torch.linalg.svd(cov_i, full_matrices=False)
        x_i, y_i = x[i] - mux_drop_i, y[i] - muy_drop_i
        vals.append(y_i @ (vh_drop_i.T @ (u_drop_i.T @ x_i)))
    return torch.stack(vals).mean()


def _truncated_brute_force_reference(
    x: torch.Tensor, y: torch.Tensor, centered: bool, k: int
) -> torch.Tensor:
    """Independent reference for the low-rank (`k`) path: truncate the full-data cross-cov to
    rank k, then leave-one-out rank-1 downdate, explicitly SVDing each ambient n_x x n_y matrix
    (no augmented-core trick). This mirrors what rank1/ab compute but forms every matrix in the
    ambient space, so it validates the `_truncate_svd`-then-`_augmented_core` reduction.

    Uses the same centering/downdate coefficients as `_prepare_xval` (denom = (m-dof)^2/m,
    scale = (m/(m-dof))^2), since the k-path shares that preprocessing; well-defined only for
    k <= min(n_x, n_y, m-1-dof)
    """
    m, nx = x.shape
    ny = y.shape[1]
    dof = 1 if centered else 0
    mux = x.mean(0) if centered else torch.zeros(nx, dtype=x.dtype, device=x.device)
    muy = y.mean(0) if centered else torch.zeros(ny, dtype=y.dtype, device=y.device)
    xc, yc = x - mux, y - muy
    cov = (xc.T @ yc) / (m - dof)
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    cov_k = (U * S) @ Vh
    denom = (m - dof) ** 2 / m
    scale = m**2 / (m - dof) ** 2
    vals = []
    for i in range(m):
        cov_downdate = cov_k - xc[i][:, None] * yc[i][None, :] / denom
        u_i, _, vh_i = torch.linalg.svd(cov_downdate, full_matrices=False)
        vals.append(scale * yc[i] @ (vh_i.T @ (u_i.T @ xc[i])))
    return torch.stack(vals).mean()


# Shapes exercised by the independent-reference tests: small M (large m/(m-1) correction),
# square, wide/tall n_x != n_y in both orientations, and two larger-n cases (n_x=100 and a
# 40x30 block) kept in the well-posed m > n regime (min(n_x,n_y) <= M-1-dof).
_REF_SHAPES = [
    (12, 5, 6),
    (20, 7, 7),
    (8, 9, 4),
    (6, 4, 5),
    (60, 100, 20),
    (45, 40, 30),
]

# (M, n_x, n_y, k) for the low-rank path. k stays <= min(n_x, n_y, M-1-dof) so the leave-one-out
# downdated cross-cov is full rank and the polar-factor estimand is well-defined. The (40,100,60)
# row is the large-n case, which is well-posed *only* under truncation to k < M.
_K_TRUNC_CASES = [(20, 7, 7, 3), (12, 8, 5, 4), (40, 100, 60, 20), (30, 12, 8, 6)]


class TestLinalgUtils(unittest.TestCase):
    def test_exact_methods_match_recompute_loo_reference(self):
        """Validate every *exact* method (not the 'orthogonal' method) vs an independent
        recompute-from-scratch LOO reference calculation.
        """
        for dt in [torch.float64]:
            for device in DEVICES:
                for centered in [False, True]:
                    for m, nx, ny in _REF_SHAPES:
                        for method in ["brute_force", "rank1", "ab", "secular"]:
                            with self.subTest(
                                msg=f"dtype={dt} device={device} centered={centered} "
                                f"shape=({m},{nx},{ny}) method={method}"
                            ):
                                torch.manual_seed(0)
                                x = torch.rand(m, nx, dtype=dt, device=device)
                                y = torch.rand(m, ny, dtype=dt, device=device)
                                ref = _recompute_loo_reference(x, y, centered)
                                got = xval_nuc_norm_cross_cov(
                                    x, y, centered=centered, method=method
                                )
                                assert_close(got, ref)

    def test_orthogonalize_kernel_matches_exact_polar(self):
        """Decoupled kernel test: the fast Newton-Schulz `orthogonalize` vs the exact polar
        factor U V^T, on a variety of matrices (batched; square / tall / wide; a range of
        condition numbers). This isolates the *approximation quality of the kernel* from the
        estimator logic -- the estimator wiring is already checked at machine precision via the
        exact rank1/ab methods, so this test owns the single "close-enough" claim.

        The tuned coefficients have a wide basin: measured element-wise error stays ~4e-3 for
        singular values down to ~1/50 of the largest (see code/measure_ortho_band.py), so a fixed
        atol=1e-2 is comfortable and dtype-independent. NS cannot recover singular values that are
        tiny relative to the largest, so inputs are conditioned into [1/cond, 1] with cond <= 20.
        """
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                for m, n in [(6, 6), (8, 5), (5, 8)]:
                    for cond in [1.0, 5.0, 20.0]:
                        with self.subTest(
                            msg=f"dtype={dt} device={device} shape=({m},{n}) cond={cond}"
                        ):
                            torch.manual_seed(0)
                            r = min(m, n)
                            a = torch.randn(32, m, n, dtype=dt, device=device)
                            u_a, _, vh_a = torch.linalg.svd(a, full_matrices=False)
                            svals = torch.linspace(1.0 / cond, 1.0, r, dtype=dt, device=device)
                            mats = (u_a * svals.unsqueeze(0)) @ vh_a
                            approx = orthogonalize(mats)
                            exact = _exact_polar(mats)
                            assert_close(approx, exact, atol=ORTHO_ATOL, rtol=ORTHO_RTOL)

    def test_orthogonalize_matches_recompute_loo_reference(self):
        """Same as test_exact_methods_match_recompute_loo_reference but for 'orthogonalize' method
        and larger closeness tolerances.
        """
        for dt in [torch.float64]:
            for device in DEVICES:
                for centered in [False, True]:
                    with self.subTest(msg=f"dtype={dt} device={device} centered={centered}"):
                        torch.manual_seed(0)
                        x = torch.rand(20, 5, dtype=dt, device=device)
                        y = torch.rand(20, 6, dtype=dt, device=device)
                        ref = _recompute_loo_reference(x, y, centered)
                        got = xval_nuc_norm_cross_cov(
                            x, y, centered=centered, method="orthogonalize"
                        )
                        assert_close(got, ref, atol=ORTHO_ATOL, rtol=ORTHO_RTOL)

    def test_k_truncation_matches_truncated_brute_force(self):
        """Low-rank path: with `k` set, the exact methods (rank1, ab) must equal an independent
        truncated brute force that forms every ambient downdated matrix explicitly, and the
        approximate orthogonalize must land within its NS band. Covers square, wide/tall, and a
        large-n (n_x=100) case, always with k in the well-posed range k <= min(n_x,n_y,M-1-dof)
        (see code/verify_k_truncation.py for the NumPy proof and the ill-posed boundary).
        """
        for dt in [torch.float64]:
            for device in DEVICES:
                for centered in [False, True]:
                    for m, nx, ny, k in _K_TRUNC_CASES:
                        torch.manual_seed(0)
                        x = torch.rand(m, nx, dtype=dt, device=device)
                        y = torch.rand(m, ny, dtype=dt, device=device)
                        ref = _truncated_brute_force_reference(x, y, centered, k)
                        for method in ["rank1", "ab", "secular"]:
                            with self.subTest(
                                msg=f"centered={centered} shape=({m},{nx},{ny}) k={k} "
                                f"method={method}"
                            ):
                                got = xval_nuc_norm_cross_cov(
                                    x, y, centered=centered, method=method, k=k
                                )
                                assert_close(got, ref)
                        with self.subTest(
                            msg=f"centered={centered} shape=({m},{nx},{ny}) k={k} "
                            f"method=orthogonalize"
                        ):
                            got = xval_nuc_norm_cross_cov(
                                x, y, centered=centered, method="orthogonalize", k=k
                            )
                            assert_close(got, ref, atol=ORTHO_ATOL, rtol=ORTHO_RTOL)

    def test_k_truncation_reduces_to_full_at_max_k(self):
        """Passing k >= the working rank must be a no-op: `_truncate_svd` only truncates when
        k < r, so k=min(n_x,n_y) (full rank here, since m > n) equals the untruncated result.
        """
        for method in ["rank1", "ab", "orthogonalize", "secular"]:
            for centered in [False, True]:
                with self.subTest(msg=f"method={method} centered={centered}"):
                    torch.manual_seed(0)
                    x = torch.rand(20, 5, dtype=torch.float64)
                    y = torch.rand(20, 6, dtype=torch.float64)
                    full = xval_nuc_norm_cross_cov(x, y, centered=centered, method=method)
                    at_k = xval_nuc_norm_cross_cov(x, y, centered=centered, method=method, k=5)
                    assert_close(full, at_k)

    def test_rank_deficient_inputs(self):
        """Near-degenerate inputs where the held-out sample's residual orthogonal to the singular
        subspace is ~0 (small alpha/beta): X and Y each live in a low-dimensional subspace, so the
        augmented (r+1)-th coordinate nearly vanishes. The estimators must stay equal to the
        independent recompute-from-scratch LOO reference (the `_augmented_core` docstring claims
        alpha=0 is benign; this exercises it end-to-end through the centering pipeline).
        """
        for method in ["brute_force", "rank1", "ab"]:
            for centered in [False, True]:
                with self.subTest(msg=f"method={method} centered={centered}"):
                    torch.manual_seed(0)
                    # Ambient n_x=8, n_y=7 but data confined to a rank-3 subspace -> residuals ~0.
                    latent = torch.rand(15, 3, dtype=torch.float64)
                    x = latent @ torch.rand(3, 8, dtype=torch.float64)
                    y = latent @ torch.rand(3, 7, dtype=torch.float64)
                    ref = _recompute_loo_reference(x, y, centered)
                    got = xval_nuc_norm_cross_cov(x, y, centered=centered, method=method)
                    assert_close(got, ref)

    def test_xcov_norm_rank1(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    result_rank1 = xval_nuc_norm_cross_cov(x, y, centered=True, method="rank1")
                    assert_close(result_brute_force, result_rank1)

                    result_rank1_flipped = xval_nuc_norm_cross_cov(
                        y, x, centered=True, method="rank1"
                    )
                    assert_close(result_rank1, result_rank1_flipped)

    def test_xcov_norm_rank1_streaming(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += xval_nuc_norm_cross_cov(b_x, b_y, method="rank1", stats=stats) / 4
                    assert_close(result_brute_force, avg)

    def test_xcov_norm_ab(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    result_ab = xval_nuc_norm_cross_cov(x, y, centered=True, method="ab")
                    assert_close(result_brute_force, result_ab)

                    result_ab_flipped = xval_nuc_norm_cross_cov(y, x, centered=True, method="ab")
                    assert_close(result_ab, result_ab_flipped)

    def test_xcov_norm_ab_streaming(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += xval_nuc_norm_cross_cov(b_x, b_y, method="ab", stats=stats) / 4
                    assert_close(result_brute_force, avg)

    def test_xcov_norm_secular(self):
        """The secular / Cauchy (bilinear-collapse) method is exact up to floating point on
        well-separated spectra: it must match brute_force, and be symmetric under X<->Y. Random
        `rand` inputs give generically distinct singular values, so the pole denominators stay
        well away from zero (the conditioning caveat only bites on clustered/coincident spectra;
        see xval_nuc_norm_cross_cov_secular docstring and notes/fmm_bilinear_analysis.md).
        """
        for m in [20, 100, 200, 1000]:
            for dt in [torch.float32, torch.float64]:
                for device in DEVICES:
                    with self.subTest(msg=f"dtype={dt} device={device} m={m}"):
                        x = torch.rand(m, 5, dtype=dt, device=device)
                        y = torch.rand(m, 6, dtype=dt, device=device)

                        result_brute_force = xval_nuc_norm_cross_cov(
                            x, y, centered=True, method="brute_force"
                        )
                        result_secular = xval_nuc_norm_cross_cov(
                            x, y, centered=True, method="secular"
                        )
                        assert_close(result_brute_force, result_secular)

                        result_secular_flipped = xval_nuc_norm_cross_cov(
                            y, x, centered=True, method="secular"
                        )
                        assert_close(result_secular, result_secular_flipped)

    def test_xcov_norm_secular_streaming(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += xval_nuc_norm_cross_cov(b_x, b_y, method="secular", stats=stats) / 4
                    assert_close(result_brute_force, avg)

    def test_xcov_norm_orthogonalize(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    result_orthogonalize = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="orthogonalize"
                    )
                    # NOTE: orthogonalization is approximate -> fixed absolute tolerance (not
                    # eps-scaled; see assert_close docstring). The X<->Y symmetry check below
                    # compares two orthogonalize outputs, which are transpose-identical up to
                    # machine precision, so it keeps the tight default tolerance.
                    assert_close(
                        result_brute_force,
                        result_orthogonalize,
                        atol=ORTHO_ATOL,
                        rtol=ORTHO_RTOL,
                    )

                    result_orthogonalize_flipped = xval_nuc_norm_cross_cov(
                        y, x, centered=True, method="orthogonalize"
                    )
                    assert_close(result_orthogonalize, result_orthogonalize_flipped)

    def test_xcov_norm_orthogonalize_streaming(self):
        for dt in [torch.float32, torch.float64]:
            for device in DEVICES:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += (
                            xval_nuc_norm_cross_cov(b_x, b_y, method="orthogonalize", stats=stats)
                            / 4
                        )
                    # NOTE: orthogonalization is approximate -> fixed absolute tolerance
                    # (dtype-independent; see assert_close docstring).
                    assert_close(result_brute_force, avg, atol=ORTHO_ATOL, rtol=ORTHO_RTOL)
