import torch
from torch.testing import assert_close
from itertools import product
from nn_lib.analysis.similarity.cka import HSICEstimator, hsic, chunsum, default_linear_kernel
import unittest


class TestHSICEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = 12
        cls.x = torch.randn(cls.m, 5, dtype=torch.float64)
        cls.y = torch.randn(cls.m, 6, dtype=torch.float64)

    def test_chunsum_xx(self):
        expected_ijlm_xx = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_ijlm_xx += torch.sum(self.x[i, :] * self.x[j, :]) * torch.sum(
                self.x[l, :] * self.x[m, :]
            ) - torch.sum(self.x[i, :] * self.x[j, :] * self.x[l, :] * self.x[m, :])
        assert_close(expected_ijlm_xx, chunsum("ijlm", self.x, self.x, is_self=True))

        expected_ijjl_xx = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_ijjl_xx += torch.sum(self.x[i, :] * self.x[j, :]) * torch.sum(
                self.x[j, :] * self.x[l, :]
            ) - torch.sum(self.x[i, :] * self.x[j, :] * self.x[j, :] * self.x[l, :])
        assert_close(expected_ijjl_xx, chunsum("ijjl", self.x, self.x, is_self=True))

        expected_iiij_xx = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_iiij_xx += torch.sum(self.x[i, :] * self.x[i, :]) * torch.sum(
                self.x[i, :] * self.x[j, :]
            ) - torch.sum(self.x[i, :] * self.x[i, :] * self.x[i, :] * self.x[j, :])
        assert_close(expected_iiij_xx, chunsum("iiij", self.x, self.x, is_self=True))

    def test_chunsum_xy(self):
        expected_ijlm_xy = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_ijlm_xy += torch.sum(self.x[i, :] * self.x[j, :]) * torch.sum(
                self.y[l, :] * self.y[m, :]
            )
        assert_close(expected_ijlm_xy, chunsum("ijlm", self.x, self.y, is_self=False))

        expected_ijjl_xy = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_ijjl_xy += torch.sum(self.x[i, :] * self.x[j, :]) * torch.sum(
                self.y[j, :] * self.y[l, :]
            )
        assert_close(expected_ijjl_xy, chunsum("ijjl", self.x, self.y, is_self=False))

        expected_iiij_xy = 0.0
        for i, j, l, m in product(range(self.m), range(self.m), range(self.m), range(self.m)):
            expected_iiij_xy += torch.sum(self.x[i, :] * self.x[i, :]) * torch.sum(
                self.x[i, :] * self.x[j, :]
            )
        assert_close(expected_iiij_xy, chunsum("iiij", self.x, self.x, is_self=False))

    def test_estimators_no_kernel(self):
        for est in HSICEstimator:
            with self.subTest(msg=str(est)):
                result = hsic(self.x, self.y, kernel_x=None, kernel_y=None, estimator=est)
                assert result.shape == torch.Size([])

                if est != HSICEstimator.CHUN2025:
                    result_explicit_linear = hsic(
                        self.x,
                        self.y,
                        kernel_x=default_linear_kernel,
                        kernel_y=default_linear_kernel,
                        estimator=est,
                    )
                    assert_close(result, result_explicit_linear)

    def test_chun_estimator_with_kernel_raises_error(self):
        with self.assertRaises(ValueError):
            hsic(
                self.x,
                self.y,
                kernel_x=default_linear_kernel,
                kernel_y=default_linear_kernel,
                estimator=HSICEstimator.CHUN2025,
            )
