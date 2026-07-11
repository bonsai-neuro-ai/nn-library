import unittest

import numpy as np
import torch
from torch.testing import assert_close as assert_close_torch

from nn_lib.utils import RunningCovariance
from nn_lib.utils.xval_nuc_norm import XValStats, xval_nuc_norm_cross_cov


def assert_close(x, y, lenience=0.0):
    atol = (10.0**lenience) * np.sqrt(torch.finfo(x.dtype).eps)
    rtol = atol / 10
    assert_close_torch(x, y, rtol=rtol, atol=atol)


class TestLinalgUtils(unittest.TestCase):
    def test_xcov_norm_rank1(self):
        for dt in [torch.float32, torch.float64]:
            for device in ["cpu", "cuda"]:
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
            for device in ["cpu", "cuda"]:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True, scalar=False)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += xval_nuc_norm_cross_cov(b_x, b_y, method="rank1", stats=stats) / 4
                    assert_close(result_brute_force, avg)

    def test_xcov_norm_ab(self):
        for dt in [torch.float32, torch.float64]:
            for device in ["cpu", "cuda"]:
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
            for device in ["cpu", "cuda"]:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True, scalar=False)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += xval_nuc_norm_cross_cov(b_x, b_y, method="ab", stats=stats) / 4
                    assert_close(result_brute_force, avg)

    def test_xcov_norm_orthogonalize(self):
        for dt in [torch.float32, torch.float64]:
            for device in ["cpu", "cuda"]:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    result_orthogonalize = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="orthogonalize"
                    )
                    # NOTE: orthogonalization is not exact, so we use a looser tolerance for this test
                    assert_close(result_brute_force, result_orthogonalize, lenience=1)

                    result_orthogonalize_flipped = xval_nuc_norm_cross_cov(
                        y, x, centered=True, method="orthogonalize"
                    )
                    assert_close(result_orthogonalize, result_orthogonalize_flipped)

    def test_xcov_norm_orthogonalize_streaming(self):
        for dt in [torch.float32, torch.float64]:
            for device in ["cpu", "cuda"]:
                with self.subTest(msg=f"dtype={dt} device={device}"):
                    x = torch.rand(20, 5, dtype=dt, device=device)
                    y = torch.rand(20, 6, dtype=dt, device=device)

                    result_brute_force = xval_nuc_norm_cross_cov(
                        x, y, centered=True, method="brute_force"
                    )
                    rc = RunningCovariance(centered=True, scalar=False)
                    rc.update(x, y)
                    stats = XValStats.from_running_covariance(rc)
                    avg = 0
                    for b_x, b_y in zip(x.reshape(4, 5, 5), y.reshape(4, 5, 6)):
                        avg += (
                            xval_nuc_norm_cross_cov(b_x, b_y, method="orthogonalize", stats=stats)
                            / 4
                        )
                    # NOTE: orthogonalization is not exact, so we use a looser tolerance for this test
                    assert_close(result_brute_force, avg, lenience=1)
