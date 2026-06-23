import unittest
from itertools import batched

import torch
from torch.testing import assert_close
from nn_lib.utils.stats import *


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

    def test_covs_uncentered(self):
        x = torch.rand((100, 2))
        y = torch.rand((100, 3))
        moments = calculate_moments_batchwise([(x, y)])
        covs = moments_to_covs(moments, centered=False)
        self.assertEqual(len(covs), 3)
        assert_close(covs["cov_0_0"], moments["moment2_0_0"].avg)
        assert_close(covs["cov_1_1"], moments["moment2_1_1"].avg)
        assert_close(covs["cov_0_1"], moments["moment2_0_1"].avg)

    def test_covs_centered(self):
        x = torch.rand((100, 2))
        y = torch.rand((100, 3))
        moments = calculate_moments_batchwise([(x, y)])
        covs = moments_to_covs(moments, centered=True)
        true_cov = torch.cov(torch.hstack([x, y]).T, correction=0)
        self.assertEqual(len(covs), 3)
        assert_close(covs["cov_0_0"], true_cov[:2, :2])
        assert_close(covs["cov_1_1"], true_cov[2:, 2:])
        assert_close(covs["cov_0_1"], true_cov[:2, 2:])
