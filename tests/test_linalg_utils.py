import unittest

import torch
from torch.testing import assert_close

from nn_lib.utils import rank_one_svd_update, xval_nuc_norm_cross_cov


class TestLinalgUtils(unittest.TestCase):
    def _test_svd_update_helper(self, x, y):
        full_mat = x.T @ y
        true_u, true_s, true_vh = torch.linalg.svd(full_mat, full_matrices=False)
        assert_close(full_mat, true_u @ torch.diag(true_s) @ true_vh)

        partial_mat = x[:-1].T @ y[:-1]
        part_u, part_s, part_vh = torch.linalg.svd(partial_mat, full_matrices=False)
        assert_close(partial_mat, part_u @ torch.diag(part_s) @ part_vh)

        # Expect SVD is *not* close initially
        with self.assertRaises(AssertionError):
            assert_close(full_mat, part_u @ torch.diag(part_s) @ part_vh)

        # Do the rank-1 SVD update
        updt_u, updt_s, updt_vh = rank_one_svd_update(part_u, part_s, part_vh, x[-1], y[-1])

        # Assert expected shapes and orthonormality properties
        self.assertEqual(updt_u.shape, true_u.shape)
        self.assertEqual(updt_s.shape, true_s.shape)
        self.assertEqual(updt_vh.shape, true_vh.shape)
        assert_close(updt_u @ updt_u.T, torch.eye(updt_u.shape[0], dtype=x.dtype))
        assert_close(updt_vh @ updt_vh.T, torch.eye(updt_vh.shape[0], dtype=x.dtype))

        # Assert SVD worked
        assert_close(full_mat, updt_u @ torch.diag(updt_s) @ updt_vh)

    def test_rank1_svd_update_full_rank(self):
        for dt in [torch.float32, torch.float64]:
            with self.subTest(msg=f"dtype={dt}"):
                self._test_svd_update_helper(
                    torch.rand(20, 5, dtype=dt),
                    torch.rand(20, 6, dtype=dt),
                )

    def test_rank1_svd_update_rank_deficient(self):
        for dt in [torch.float32, torch.float64]:
            with self.subTest(msg=f"dtype={dt}"):
                self._test_svd_update_helper(
                    torch.rand(4, 5, dtype=dt),
                    torch.rand(4, 6, dtype=dt),
                )

    def test_xcov_norm_rank1(self):
        for dt in [torch.float32, torch.float64]:
            with self.subTest(msg=f"dtype={dt}"):
                x = torch.rand(20, 5, dtype=dt)
                y = torch.rand(20, 6, dtype=dt)

                result_brute_force = xval_nuc_norm_cross_cov(x, y, method="brute_force")
                result_rank1 = xval_nuc_norm_cross_cov(x, y, method="rank1")

                assert_close(result_brute_force, result_rank1)

    def test_xcov_norm_ab(self):
        for dt in [torch.float32, torch.float64]:
            with self.subTest(msg=f"dtype={dt}"):
                x = torch.rand(20, 5, dtype=dt)
                y = torch.rand(20, 6, dtype=dt)

                result_brute_force = xval_nuc_norm_cross_cov(x, y, method="brute_force")
                result_ab = xval_nuc_norm_cross_cov(x, y, method="ab")

                assert_close(result_brute_force, result_ab)

    def test_xcov_norm_orthogonalize(self):
        for dt in [torch.float32, torch.float64]:
            with self.subTest(msg=f"dtype={dt}"):
                x = torch.rand(20, 5, dtype=dt)
                y = torch.rand(20, 6, dtype=dt)

                result_brute_force = xval_nuc_norm_cross_cov(x, y, method="brute_force")
                result_orthogonalize = xval_nuc_norm_cross_cov(x, y, method="orthogonalize")

                # NOTE: orthogonalization is not exact, so we use a looser tolerance for this test
                assert_close(result_brute_force, result_orthogonalize, rtol=3e-3, atol=3e-3)
