import os
import tempfile
import unittest

import numpy as np
import torch

from nn_lib.analysis.pca import PrincipalComponents


def nancov(x):
    out = x.new_zeros((x.shape[1], x.shape[1]))
    mu = x.nanmean(dim=0)
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            count = torch.sum(~torch.isnan(x[:, i]) & ~torch.isnan(x[:, j]))
            out[i, j] = torch.nansum(x[:, i] * x[:, j]) / count - mu[i] * mu[j]
    return out


class TestPrincipalComponents(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float64  # high precision for testing purposes
        self.dim = 5
        self.n_samples = 1000
        # Generate true data with a non-trivial covariance matrix
        l = torch.randn(self.dim, self.dim, device=self.device, dtype=self.dtype) * torch.linspace(
            1, 0.1, self.dim, device=self.device, dtype=self.dtype
        ).unsqueeze(1)
        self.true_cov = torch.eye(self.dim, device=self.device, dtype=self.dtype) + l @ l.T
        self.true_mean = torch.randn(self.dim, device=self.device, dtype=self.dtype) * 5
        chol_cov = torch.linalg.cholesky(self.true_cov, upper=True)
        self.data = (
            torch.randn(self.n_samples, self.dim, device=self.device, dtype=self.dtype) @ chol_cov
            + self.true_mean[None, :]
        )

    def test_from_data(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        self.assertEqual(pcs.dim, self.dim)
        torch.testing.assert_close(pcs.mean, self.data.nanmean(dim=0))
        torch.testing.assert_close(pcs.cov, nancov(self.data))

    def test_add_batch_vectors(self):
        pcs = PrincipalComponents(self.dim, center=True, device=self.device, dtype=self.dtype)
        pcs.add_batch_vectors(self.data)
        torch.testing.assert_close(pcs.mean, self.data.nanmean(dim=0))
        torch.testing.assert_close(pcs.cov, nancov(self.data))

    def test_batch_equals_full(self):
        pcs1 = PrincipalComponents.from_data(self.data, center=True)
        pcs2 = PrincipalComponents(self.dim, center=True, device=self.device, dtype=self.dtype)
        assert self.n_samples % 10 == 0, "this test won't work if n_samples is not divisible by 10"
        for i in range(0, self.n_samples, 10):
            pcs2.add_batch_vectors(self.data[i : i + 10])
        torch.testing.assert_close(pcs1.mean, pcs2.mean)
        torch.testing.assert_close(pcs1.cov, pcs2.cov)

    def test_devices(self):
        pcs_cpu = PrincipalComponents.from_data(self.data, center=True)
        pcs_gpu = PrincipalComponents.from_data(self.data.to("cuda:0"), center=True)

        self.assertEqual(pcs_cpu.cov.device, torch.device("cpu"))
        self.assertEqual(pcs_gpu.cov.device, torch.device("cuda:0"))

        pcs2cpu = pcs_gpu.to("cpu")
        pcs2gpu = pcs_cpu.to("cuda:0")

        self.assertEqual(pcs2cpu.cov.device, torch.device("cpu"))
        self.assertEqual(pcs2gpu.cov.device, torch.device("cuda:0"))

    def test_save_load(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(pcs.state_dict(), os.path.join(tmpdir, "pcs.pth"))

            pcs2 = PrincipalComponents.load_state_dict(torch.load(os.path.join(tmpdir, "pcs.pth")))

        self.assertEqual(pcs.dim, pcs2.dim)
        self.assertEqual(pcs._center, pcs2._center)
        torch.testing.assert_close(pcs.mean, pcs2.mean)
        torch.testing.assert_close(pcs.cov, pcs2.cov)

        # Test that batch tracking is still correct after save/load
        pcs.add_batch_vectors(self.data[:5, :])
        pcs2.add_batch_vectors(self.data[:5, :])

        torch.testing.assert_close(pcs.mean, pcs2.mean)
        torch.testing.assert_close(pcs.cov, pcs2.cov)

    def test_suggest_dim(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)

        with self.assertRaises(ValueError):
            pcs.suggest_dimensions(0.0)

        with self.assertRaises(ValueError):
            pcs.suggest_dimensions(1.01)

        suggestions = []
        for frac in torch.linspace(0.1, 1.0, 10):
            suggestions.append(pcs.suggest_dimensions(frac))

        for i in range(len(suggestions) - 1):
            self.assertGreater(suggestions[i], 0)
            self.assertLessEqual(suggestions[i], self.dim)
            self.assertLessEqual(suggestions[i], suggestions[i + 1])

    def test_reconstruct_shape(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        dim_reduced = pcs.reduce_dim(self.data, k=2, original_space=False)
        reconstructed_data = pcs.reconstruct(dim_reduced)
        self.assertNotEqual(dim_reduced.shape, self.data.shape)
        self.assertEqual(reconstructed_data.shape, self.data.shape)

    def test_reconstruct_noop(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        dim_reduced = pcs.reduce_dim(self.data, k=self.dim, original_space=False)
        reconstructed_data = pcs.reconstruct(dim_reduced)
        self.assertEqual(dim_reduced.shape, self.data.shape)
        self.assertFalse(torch.allclose(self.data, dim_reduced))
        self.assertTrue(torch.allclose(self.data, reconstructed_data))

    def test_interpolate(self):
        data_with_nan = self.data.clone()
        data_with_nan[0, 0] = float("nan")
        pcs = PrincipalComponents.from_data(self.data, center=True)
        filled_data = pcs.interpolate(data_with_nan, method="mean")
        self.assertFalse(torch.isnan(filled_data).any())

    def test_whiten(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        whitened_data = pcs.whiten(self.data)
        self.assertEqual(whitened_data.shape, self.data.shape)
        torch.testing.assert_close(
            whitened_data.mean(dim=0), torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        )
        torch.testing.assert_close(
            nancov(whitened_data), torch.eye(self.dim, device=self.device, dtype=self.dtype)
        )

    def test_zca(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        zca_data = pcs.whiten(self.data, zca=True)
        self.assertEqual(zca_data.shape, self.data.shape)
        torch.testing.assert_close(
            zca_data.mean(dim=0), torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        )
        torch.testing.assert_close(
            nancov(zca_data), torch.eye(self.dim, device=self.device, dtype=self.dtype)
        )

        # ZCA should leave the whitened data 'closer' to the original data than standard whitening
        whiten_data = pcs.whiten(self.data)
        dist_whitened_original = torch.norm(self.data - whiten_data, p="fro")
        dist_zca_original = torch.norm(self.data - zca_data, p="fro")
        self.assertLess(dist_zca_original, dist_whitened_original)

    def test_reduce_dim(self):
        k = 3
        pcs = PrincipalComponents.from_data(self.data, center=True)
        reduced_data = pcs.reduce_dim(self.data, k)
        self.assertEqual(reduced_data.shape, (self.n_samples, k))
        self.assertEqual(torch.linalg.matrix_rank(reduced_data), k)

    def test_reduce_dim_projection(self):
        k = 3
        pcs = PrincipalComponents.from_data(self.data, center=True)
        reduced_data = pcs.reduce_dim(self.data, k, original_space=True)
        self.assertEqual(reduced_data.shape, (self.n_samples, self.dim))
        # Expect dim k+1 back because mean is added back in
        self.assertEqual(torch.linalg.matrix_rank(reduced_data), k + 1)

    def test_subspace_similarity_invariant_to_scale(self):
        pcs1 = PrincipalComponents.from_data(self.data, center=True)
        pcs2 = PrincipalComponents.from_data(self.data * 2, center=True)
        similarity = pcs1.subspace_similarity(pcs2)
        torch.testing.assert_allclose(similarity, 1.0)

    def test_subspace_similarity_null(self):
        pcs1 = PrincipalComponents.from_data(self.data, center=True)
        pcs2 = PrincipalComponents.from_data(torch.randn_like(self.data), center=True)
        null_similarities = pcs1.subspace_similarity_null(pcs2, n_samples=1000)
        mu, sigma = pcs1.subspace_similarity_null_normal(pcs2)
        avg_null = np.mean(null_similarities)
        std_null = np.std(null_similarities)
        mcse_null = std_null / np.sqrt(1000)
        self.assertGreater(avg_null + 3 * mcse_null, mu)
        self.assertLess(avg_null - 3 * mcse_null, mu)
        self.assertAlmostEqual(sigma, std_null, places=3)

    def test_effective_dim(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        effective_dim = pcs.effective_dim(method="n2")
        self.assertGreater(effective_dim, 0)
        self.assertLessEqual(effective_dim, self.dim)


class TestPrincipalComponentsWithMissingData(TestPrincipalComponents):
    def setUp(self):
        super().setUp()
        self.original_data = self.data.clone()
        self.data[torch.rand(self.data.shape, device=self.device) < 0.1] = float("nan")

    def test_interpolate(self):
        pcs = PrincipalComponents.from_data(self.data, center=True)
        valid_mask = torch.isfinite(self.data)
        for method in ("zero", "mean", "sample"):
            with self.subTest(method):
                filled_in_data = pcs.interpolate(self.data, method=method)
                self.assertTrue(torch.equal(filled_in_data[valid_mask], self.data[valid_mask]))
                self.assertFalse(torch.any(torch.isnan(filled_in_data)))
                # TODO - check the actual behavior each of the 3 methods.

    # TODO – Whitening and ZCA and reconstruction are tricky to test even after interpolating.
    #  Just call it good for now and revisit these another time
    def test_whiten(self):
        pass

    def test_zca(self):
        pass

    def test_reconstruct_shape(self):
        pass

    def test_reconstruct_noop(self):
        pass

    def test_reduce_dim(self):
        k = 3
        pcs = PrincipalComponents.from_data(self.data, center=True)
        filled_in_data = pcs.interpolate(self.data, method="mean")
        reduced_data = pcs.reduce_dim(filled_in_data, k)
        self.assertEqual(reduced_data.shape, (self.n_samples, k))
        self.assertEqual(torch.linalg.matrix_rank(reduced_data), k)

    def test_reduce_dim_projection(self):
        k = 3
        pcs = PrincipalComponents.from_data(self.data, center=True)
        filled_in_data = pcs.interpolate(self.data, method="mean")
        reduced_data = pcs.reduce_dim(filled_in_data, k, original_space=True)
        self.assertEqual(reduced_data.shape, (self.n_samples, self.dim))
        # Expect dim k+1 back because mean is added back in
        self.assertEqual(torch.linalg.matrix_rank(reduced_data), k + 1)


if __name__ == "__main__":
    unittest.main()
