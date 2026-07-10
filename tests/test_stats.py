import unittest
from itertools import batched

import torch
from torch.testing import assert_close

from nn_lib.utils.stats import (
    RunningAverage,
    moments_to_covs,
    calculate_moments_batchwise,
    RunningCovariance,
)


class TestStats(unittest.TestCase):
    def test_running_average_1(self):
        values = torch.rand(10)
        ra = RunningAverage()
        for v in values:
            ra.update(v, batch_count=1)
        assert_close(ra.avg, torch.mean(values))

    def test_running_average_batched(self):
        values = torch.rand(10)
        ra = RunningAverage()
        for v in batched(values, 2):
            ra.update(sum(v) / len(v), batch_count=2)
        assert_close(ra.avg, torch.mean(values))

    def test_moments(self):
        x = torch.rand((100, 2))
        y = torch.rand((100, 3))
        moments = calculate_moments_batchwise([(x, y)])
        self.assertEqual(len(moments), 5)
        assert_close(moments["moment1_0"].avg, torch.mean(x, dim=0))
        assert_close(moments["moment2_0_0"].avg, torch.einsum("ni,nj->ij", x, x) / 100)
        assert_close(moments["moment1_1"].avg, torch.mean(y, dim=0))
        assert_close(moments["moment2_1_1"].avg, torch.einsum("ni,nj->ij", y, y) / 100)
        assert_close(moments["moment2_0_1"].avg, torch.einsum("ni,nj->ij", x, y) / 100)

        # Now do it batchy
        moments = calculate_moments_batchwise([(x[:50], y[:50]), (x[50:], y[50:])])
        self.assertEqual(len(moments), 5)
        assert_close(moments["moment1_0"].avg, torch.mean(x, dim=0))
        assert_close(moments["moment2_0_0"].avg, torch.einsum("ni,nj->ij", x, x) / 100)
        assert_close(moments["moment1_1"].avg, torch.mean(y, dim=0))
        assert_close(moments["moment2_1_1"].avg, torch.einsum("ni,nj->ij", y, y) / 100)
        assert_close(moments["moment2_0_1"].avg, torch.einsum("ni,nj->ij", x, y) / 100)

    def test_cov_methods_numerical_stability(self):
        x1 = torch.rand((1000, 2))
        y1 = torch.rand((1000, 3))
        moments1 = calculate_moments_batchwise([(x1, y1)], covariances=True)
        naive_covs1 = moments_to_covs(moments1, centered=True)
        stable_covs1 = {k: v for k, v in moments1.items() if k.startswith("cov")}

        # x2 and y2 are shifted copies of x1 and y1. This makes E[x^2]-E[x]^2 numerically unstable.
        x2 = x1 + 1e3
        y2 = y1 + 1e3
        moments2 = calculate_moments_batchwise([(x2, y2)], covariances=True)
        naive_covs2 = moments_to_covs(moments2, centered=True)
        stable_covs2 = {k: v for k, v in moments2.items() if k.startswith("cov")}

        # Theoretically cov(x1,y1) == cov(x2,y2) since cov should be invariant to offsets when
        # centered. But the 'moments_to_covs' method is numerically imprecise (catastrophic
        # canceling) so we assert here that they *aren't* the same as a way of asserting that
        # numerical instability is at play. Hence the deprecation of 'moments_to_covs'.
        for cov1, cov2 in zip(naive_covs1.values(), naive_covs2.values()):
            with self.assertRaises(AssertionError):
                assert_close(cov1, cov2)

        # ...but, the Welford algorithm should have produced stable results.
        for cov1, cov2 in zip(stable_covs1.values(), stable_covs2.values()):
            assert_close(cov1.avg, cov2.avg)

    def test_running_variance(self):
        # 10 batches of 100 values each, with dimension 3
        values = torch.rand(10, 100, 3)
        rc0 = RunningCovariance(dof=0, scalar=True)
        rc1 = RunningCovariance(dof=1, scalar=True)
        for v in values:
            rc1.update(v)
            rc0.update(v)

        self.assertEqual(rc1.count, 1000)

        # In 'scalar' mode we get variances of each dimension of x
        self.assertEqual(rc1.avg.shape, (3,))

        # In 'scalar' mode we can refer to rc.variance but not rc.covariance
        _ = rc1.variance
        with self.assertRaises(ValueError):
            _ = rc1.covariance

        # Now compare results to biased (dof=0) and unbiased (dof=1) estimators.
        est_var_0 = rc0.avg
        est_var_1 = rc1.avg
        true_var_0 = torch.var(values.view(-1, 3), dim=0, unbiased=False)
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
        rc0 = RunningCovariance(dof=0, scalar=False)
        rc1 = RunningCovariance(dof=1, scalar=False)
        for v in values:
            rc1.update(v)
            rc0.update(v)

        self.assertEqual(rc1.count, 1000)

        # In 'scalar=False' mode we get the covariance matrix of columns of x
        self.assertEqual(rc1.avg.shape, (3, 3))

        # In 'scalar=False' mode we can refer to rc.covariance but not rc.variance
        _ = rc1.covariance
        with self.assertRaises(ValueError):
            _ = rc1.variance

        # Now compare results to biased (dof=0) and unbiased (dof=1) estimators.
        est_cov_0 = rc0.avg
        est_cov_1 = rc1.avg
        true_cov_0 = torch.cov(values.view(-1, 3).T, correction=0)
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

        # In 'scalar=False' mode we get the covariance matrix of columns of x
        self.assertEqual(rc_x.avg.shape, (3, 3))
        self.assertEqual(rc_xx.avg.shape, (3, 3))

        assert_close(rc_x.avg, rc_xx.avg)


    def test_running_cross_covariance(self):
        values_x = torch.rand(5, 100, 3)
        values_y = torch.rand(5, 100, 4)
        rc0 = RunningCovariance(dof=0, scalar=False)
        rc1 = RunningCovariance(dof=1, scalar=False)
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
        true_cov_0 = torch.cov(values_xy.view(-1, 7).T, correction=0)[:3, 3:]
        true_cov_1 = torch.cov(values_xy.view(-1, 7).T, correction=1)[:3, 3:]
        assert_close(actual=est_cov_1, expected=true_cov_1)
        assert_close(actual=est_cov_0, expected=true_cov_0)

        # Sanity-check for triviality: the cross-checks should *not* be equal
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_1, expected=true_cov_0)
        with self.assertRaises(AssertionError):
            assert_close(actual=est_cov_0, expected=true_cov_1)
