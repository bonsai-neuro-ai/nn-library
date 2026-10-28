import unittest
from itertools import batched

import torch
from torch.testing import assert_close

from nn_lib.utils.stats import (
    RunningAverage,
    RunningVariance,
    RunningCovariance,
)


# Only test on CUDA when a GPU is present, so the suite still passes on CPU-only machines.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def mean_helper(x, centered):
    if centered:
        return x.mean(dim=0)
    else:
        return torch.zeros_like(x.mean(dim=0))


def variance_helper(x, centered):
    if centered:
        return torch.var(x, dim=0)
    else:
        return torch.mean(x * x, dim=0)


def covariance_helper(x, y, centered):
    xy = torch.concat((x, y), dim=1)
    if centered:
        return torch.cov(xy.T)[: x.shape[1], x.shape[1] :]
    else:
        return x.T @ y / len(x)


class TestStats(unittest.TestCase):
    def setUp(self):
        # Fixed seed so the "these should NOT match" assertions can't flake.
        torch.manual_seed(0)

    def test_running_average_1(self):
        for device in DEVICES:
            with self.subTest(f"device={device}"):
                values = torch.rand(10, device=device)
                ra = RunningAverage()
                for v in values:
                    ra.update(v, batch_count=1)
                assert_close(ra.avg, torch.mean(values))

    def test_running_average_batched(self):
        for device in DEVICES:
            with self.subTest(f"device={device}"):
                values = torch.rand(10, device=device)
                ra = RunningAverage()
                for v in batched(values, 2):
                    ra.update(sum(v) / len(v), batch_count=2)
                assert_close(ra.avg, torch.mean(values))

    def test_covariance_x(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(f"device={device}, centered={centered}"):
                    x = torch.rand((100, 2), device=device)
                    rc = RunningCovariance(centered=centered)
                    rc.update(x)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(x, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, x, centered=centered))

                    # Now do it batchy
                    rc = RunningCovariance(centered=centered)
                    for batch in x.reshape(5, 20, 2):
                        rc.update(batch)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(x, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, x, centered=centered))

    def test_variance_vector_x(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(f"device={device}, centered={centered}"):
                    x = torch.rand((100, 2), device=device)
                    rv = RunningVariance(centered=centered)
                    rv.update(x)

                    assert_close(rv.mu_x, mean_helper(x, centered=centered))
                    assert_close(rv.variance, variance_helper(x, centered=centered))

                    # Now do it batchy
                    rv = RunningVariance(centered=centered)
                    for batch in x.reshape(5, 20, 2):
                        rv.update(batch)

                    self.assertEqual(rv.variance.shape, (2,))

                    assert_close(rv.mu_x, mean_helper(x, centered=centered))
                    assert_close(rv.variance, variance_helper(x, centered=centered))

    def test_variance_tensor_x(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(f"device={device}, centered={centered}"):
                    x = torch.rand((100, 2, 3, 4), device=device)
                    rv = RunningVariance(centered=centered)
                    rv.update(x)

                    assert_close(rv.mu_x, mean_helper(x, centered=centered))
                    assert_close(rv.variance, variance_helper(x, centered=centered))

                    # Now do it batchy
                    rv = RunningVariance(centered=centered)
                    for batch in x.reshape(5, 20, 2, 3, 4):
                        rv.update(batch)

                    self.assertEqual(rv.variance.shape, (2, 3, 4))

                    assert_close(rv.mu_x, mean_helper(x, centered=centered))
                    assert_close(rv.variance, variance_helper(x, centered=centered))

    def test_covariance_xx(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(f"device={device}, centered={centered}"):
                    x = torch.rand((100, 2), device=device)
                    rc = RunningCovariance(centered=centered)
                    rc.update(x, x)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(x, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, x, centered=centered))

                    # Now do it batchy
                    rc = RunningCovariance(centered=centered)
                    for batch in x.reshape(5, 20, 2):
                        rc.update(batch, batch)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(x, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, x, centered=centered))

    def test_covariance_xy(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(f"device={device}, centered={centered}"):
                    x = torch.rand((100, 2), device=device)
                    y = torch.rand((100, 2), device=device)
                    rc = RunningCovariance(centered=centered)
                    rc.update(x, y)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(y, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, y, centered=centered))

                    # Now do it batchy
                    rc = RunningCovariance(centered=centered)
                    for batch_x, batch_y in zip(x.reshape(5, 20, 2), y.reshape(5, 20, 2)):
                        rc.update(batch_x, batch_y)

                    assert_close(rc.mu_x, mean_helper(x, centered=centered))
                    assert_close(rc.mu_y, mean_helper(y, centered=centered))
                    assert_close(rc.covariance, covariance_helper(x, y, centered=centered))

    def test_cov_methods_numerical_stability(self):
        x1 = torch.rand((1000, 2))
        y1 = torch.rand((1000, 3))

        def _naive_cov(x_, y_):
            # Naive cov estimator is E[xy]-E[x]E[y]
            E_xy = x_.T @ y_ / 1000
            E_x = torch.mean(x_, dim=0)
            E_y = torch.mean(y_, dim=0)
            return E_xy - E_x[:, None] * E_y[None, :]

        # Get both the naive and stable cov estimators for x1 y1
        naive_cov_xy1 = _naive_cov(x1, y1)
        centered_moments = RunningCovariance(centered=True)
        centered_moments.update(x1, y1)
        stable_cov_xy = centered_moments.covariance

        # Assert that the naive and stable estimators agree (which is expected as long as E[x] and
        # E[y] are small)
        assert_close(naive_cov_xy1, stable_cov_xy)

        # x2 and y2 are shifted copies of x1 and y1. This is expected to make E[x^2]-E[x]^2
        # numerically unstable.
        x2 = x1 + 1e3
        y2 = y1 + 1e3
        # Get both the naive and stable cov estimators for x1 y1
        naive_cov_xy2 = _naive_cov(x2, y2)
        centered_moments = RunningCovariance(centered=True)
        centered_moments.update(x2, y2)
        stable_cov_xy2 = centered_moments.covariance

        # Assert that the naive and stable estimators no longer agree (this is the numerical
        # instability)
        with self.assertRaises(AssertionError):
            assert_close(naive_cov_xy2, stable_cov_xy2)

        # Assert that the stable estimates agree with each other
        assert_close(stable_cov_xy, stable_cov_xy2)

    def test_running_variance(self):
        # 10 batches of 100 values each, with dimension 3
        values = torch.rand(10, 100, 3)
        rv0 = RunningVariance(centered=False)
        rv1 = RunningVariance(centered=True)
        for v in values:
            rv1.update(v)
            rv0.update(v)

        self.assertEqual(rv1.count, 1000)

        # Check the 'avg' property refers to the variance
        assert_close(rv1.avg, rv1.variance)

        # Now compare results to biased (dof=0) and unbiased (dof=1) estimators.
        est_var_0 = rv0.avg
        est_var_1 = rv1.avg
        true_var_0 = torch.mean(values.view(-1, 3).pow(2), dim=0)
        true_var_1 = torch.var(values.view(-1, 3), dim=0, unbiased=True)
        assert_close(actual=est_var_1, expected=true_var_1)
        assert_close(actual=est_var_0, expected=true_var_0)

        # Sanity-check for triviality: the cross-checks should *not* be equal
        with self.assertRaises(AssertionError):
            assert_close(actual=est_var_1, expected=true_var_0)
        with self.assertRaises(AssertionError):
            assert_close(actual=est_var_0, expected=true_var_1)

    def test_running_covariance(self):
        # 10 batches of 100 values each, with dimension 3
        values = torch.rand(10, 100, 3)
        rc0 = RunningCovariance(centered=False)
        rc1 = RunningCovariance(centered=True)
        for v in values:
            rc1.update(v)
            rc0.update(v)

        self.assertEqual(rc1.count, 1000)

        # Check the 'avg' property refers to the covariance
        assert_close(rc1.avg, rc1.covariance)

        # Now compare results to biased (dof=0) and unbiased (dof=1) estimators.
        est_cov_0 = rc0.avg
        est_cov_1 = rc1.avg
        true_cov_0 = torch.mean(values.view(-1, 1, 3) * values.view(-1, 3, 1), dim=0)
        true_cov_1 = torch.cov(values.view(-1, 3).T, correction=1)
        assert_close(actual=est_cov_1, expected=true_cov_1)
        assert_close(actual=est_cov_0, expected=true_cov_0)

        # Sanity-check for triviality: the cross-checks should *not* be equal
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_1, expected=true_cov_0)
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_0, expected=true_cov_1)

    def test_running_covariance_x_x(self):
        # 10 batches of 100 values each, with dimension 3
        values = torch.rand(10, 100, 3)
        rc_x = RunningCovariance()
        rc_xx = RunningCovariance()
        for v in values:
            rc_x.update(v)
            rc_xx.update(v, v)

        self.assertEqual(rc_x.count, 1000)
        self.assertEqual(rc_xx.count, 1000)

        # 'avg' alias test
        assert_close(rc_x.avg, rc_x.covariance)
        assert_close(rc_xx.avg, rc_xx.covariance)

        # 'avg' value test
        assert_close(rc_x.avg, rc_xx.avg)

    def test_running_cross_covariance(self):
        values_x = torch.rand(5, 100, 3)
        values_y = torch.rand(5, 100, 4)
        rc0 = RunningCovariance(centered=False)
        rc1 = RunningCovariance(centered=True)
        for v_x, v_y in zip(values_x, values_y):
            rc1.update(v_x, v_y)
            rc0.update(v_x, v_y)

        self.assertEqual(rc1.count, 500)

        # Cross-covariance with x (dim 3) along rows and y (dim 4) along columns
        self.assertEqual(rc1.avg.shape, (3, 4))

        # Now compare results to biased (dof=0) and unbiased (dof=1) estimators.
        est_cov_0 = rc0.avg
        est_cov_1 = rc1.avg
        values_xy = torch.concat([values_x, values_y], dim=-1)
        true_cov_0 = torch.mean(values_x.view(-1, 3, 1) * values_y.view(-1, 1, 4), dim=0)
        true_cov_1 = torch.cov(values_xy.view(-1, 7).T, correction=1)[:3, 3:]
        assert_close(actual=est_cov_1, expected=true_cov_1)
        assert_close(actual=est_cov_0, expected=true_cov_0)

        # Sanity-check for triviality: the cross-checks should *not* be equal
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_1, expected=true_cov_0)
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_0, expected=true_cov_1)


def split_uneven(x, sizes):
    """Split x along dim 0 into chunks of the given sizes (must sum to len(x))."""
    assert sum(sizes) == len(x)
    return list(torch.split(x, sizes))


# Uneven sizes, including size-1 batches at the start and in the middle (the b=1 / Welford case
# of the merge), and a batch larger than everything seen so far (b > m).
UNEVEN_SIZES = [1, 7, 1, 30, 61]


class TestBatchedUpdates(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_running_average_unequal_batches(self):
        # With equal batch sizes, a bug that ignored batch_count would still give the right answer.
        values = torch.rand(100)
        ra = RunningAverage()
        for chunk in split_uneven(values, UNEVEN_SIZES):
            ra.update(chunk.mean(), batch_count=len(chunk))
        self.assertEqual(ra.count, 100)
        assert_close(ra.avg, values.mean())

    def test_variance_uneven_batches(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(device=device, centered=centered):
                    x = torch.rand((100, 3, 2), device=device)
                    rv = RunningVariance(centered=centered)
                    for chunk in split_uneven(x, UNEVEN_SIZES):
                        rv.update(chunk)
                    self.assertEqual(rv.count, 100)
                    assert_close(rv.mu_x, mean_helper(x, centered))
                    assert_close(rv.variance, variance_helper(x, centered))

    def test_covariance_uneven_batches(self):
        for device in DEVICES:
            for centered in [False, True]:
                with self.subTest(device=device, centered=centered):
                    x = torch.rand((100, 3), device=device)
                    y = torch.rand((100, 4), device=device)
                    rc = RunningCovariance(centered=centered)
                    for cx, cy in zip(split_uneven(x, UNEVEN_SIZES), split_uneven(y, UNEVEN_SIZES)):
                        rc.update(cx, cy)
                    self.assertEqual(rc.count, 100)
                    assert_close(rc.mu_x, mean_helper(x, centered))
                    assert_close(rc.mu_y, mean_helper(y, centered))
                    assert_close(rc.covariance, covariance_helper(x, y, centered))

    def test_covariance_all_size_one_batches(self):
        # Pure Welford limit of the batched update: every batch has b=1.
        x = torch.rand((50, 3))
        rc = RunningCovariance()
        for row in x:
            rc.update(row[None])
        assert_close(rc.covariance, covariance_helper(x, x, True))

    def test_covariance_tensor_valued_inputs_are_flattened(self):
        for centered in [False, True]:
            with self.subTest(centered=centered):
                x = torch.rand((60, 2, 3))
                y = torch.rand((60, 4, 1))
                rc = RunningCovariance(centered=centered)
                for cx, cy in zip(x.split(20), y.split(20)):
                    rc.update(cx, cy)
                self.assertEqual(rc.covariance.shape, (6, 4))
                self.assertEqual(rc.mu_x.shape, (6,))
                xf, yf = x.reshape(60, -1), y.reshape(60, -1)
                assert_close(rc.covariance, covariance_helper(xf, yf, centered))

    def test_iterable_inputs_match_tensor_inputs(self):
        # The docstrings promise a list of per-item tensors also works.
        x = torch.rand((40, 3))
        y = torch.rand((40, 2))
        rc_t, rc_l = RunningCovariance(), RunningCovariance()
        rv_t, rv_l = RunningVariance(), RunningVariance()
        for cx, cy in zip(x.split(10), y.split(10)):
            rc_t.update(cx, cy)
            rc_l.update(list(cx), list(cy))
            rv_t.update(cx)
            rv_l.update(list(cx))
        assert_close(rc_l.covariance, rc_t.covariance)
        assert_close(rv_l.variance, rv_t.variance)

    def test_empty_batch_is_noop(self):
        x = torch.rand((20, 3))
        rc, rv = RunningCovariance(), RunningVariance()
        rc.update(x)
        rv.update(x)
        rc.update(torch.empty(0, 3))
        rc.update([])
        rv.update(torch.empty(0, 3))
        self.assertEqual(rc.count, 20)
        self.assertEqual(rv.count, 20)
        assert_close(rc.covariance, covariance_helper(x, x, True))

    def test_no_data_raises(self):
        for cls in [RunningVariance, RunningCovariance]:
            for centered in [False, True]:
                with self.subTest(cls=cls.__name__, centered=centered):
                    obj = cls(centered=centered)
                    with self.assertRaises(ValueError):
                        _ = obj.avg
                    with self.assertRaises(ValueError):
                        _ = obj.mu_x

    def test_mismatched_batch_sizes_raise(self):
        # (10, 4) vs (20, 2) have the same numel, so a reshape-before-check bug would slip through.
        rc = RunningCovariance()
        with self.assertRaises(ValueError):
            rc.update(torch.rand(10, 4), torch.rand(20, 2))

    def test_dtype_and_device_preserved(self):
        for device in DEVICES:
            with self.subTest(device=device):
                x = torch.rand((30, 3), dtype=torch.float64, device=device)
                rc = RunningCovariance()
                rc.update(x[:10])
                rc.update(x[10:])
                self.assertEqual(rc.covariance.dtype, torch.float64)
                self.assertEqual(rc.covariance.device.type, device)

    def test_inputs_not_mutated(self):
        x = torch.rand((20, 3))
        x_copy = x.clone()
        rc, rv = RunningCovariance(), RunningVariance()
        for chunk in x.split(5):
            rc.update(chunk)
            rv.update(chunk)
        assert_close(x, x_copy)

    def test_stability_across_batches_with_drift(self):
        # A large shared offset plus a mean that drifts from batch to batch; this exercises the
        # mb/(m+b) merge term, which a single update() call never touches.
        base = torch.rand((1000, 3), dtype=torch.float64)
        drift = torch.arange(10, dtype=torch.float64).repeat_interleave(100)[:, None] * 0.1
        x32 = (base + drift + 1e4).float()
        # Reference from the *same* float32 values (casting to float32 at a 1e4 offset already
        # quantizes the data), computed in float64.
        expected = covariance_helper(x32.double(), x32.double(), True)

        rc = RunningCovariance()
        for chunk in x32.split(100):
            rc.update(chunk)
        # float32 spacing at 1e4 is ~1e-3, so ~1e-3 absolute error is the floor for any method.
        tol = dict(rtol=1e-2, atol=1e-3)
        assert_close(rc.covariance.double(), expected, **tol)

        # ...while the naive E[xy] - E[x]E[y] estimator is off by orders of magnitude.
        n = len(x32)
        naive = (x32.T @ x32 / n - x32.mean(0)[:, None] * x32.mean(0)[None, :]) * n / (n - 1)
        with self.assertRaises(AssertionError):
            assert_close(naive.double(), expected, **tol)
