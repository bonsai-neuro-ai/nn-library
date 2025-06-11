import warnings
from typing import Self, assert_never, Literal

import torch
from tqdm.auto import trange


class PrincipalComponents(object):
    """Class to help tracking dominant modes of variability of some vectors, compare subspaces, etc.

    Constructor args:
    - dim: int, dimensionality of the vectors
    - center: bool, if True, data are assumed to have a nonzero mean calculated from the empirical
        first moment. If False, data are assumed to have zero mean and the covariance is set to the
        empirical second moment.
    """

    @staticmethod
    def from_data(vecs: torch.Tensor, center: bool = True) -> "PrincipalComponents":
        """Construct a PrincipalComponents object from a matrix of vectors."""
        dim = vecs.shape[1]
        pcs = PrincipalComponents(dim, center=center, device=vecs.device, dtype=vecs.dtype)
        pcs.add_batch_vectors(vecs)
        return pcs

    def __init__(
        self,
        dim: int,
        center: bool = True,
        device: str | torch.device = "cpu",
        dtype=torch.float32,
    ):
        self.dim = dim
        self.moment1 = torch.zeros(dim, device=device, dtype=dtype)
        self.moment2 = torch.zeros(dim, dim, device=device, dtype=dtype)
        self._center = center
        self._svd_cache = None
        # In case of missing data, need to count number of samples *per dimension* of moment1 and
        # moment2, rather than just tracking the total number of samples.
        self._count_m1 = torch.zeros(1, device=device)
        self._count_m2 = torch.zeros(1, device=device)

    def to(self, device: str | torch.device) -> Self:
        """Move the object to a different device."""
        self.moment1 = self.moment1.to(device)
        self.moment2 = self.moment2.to(device)
        self._count_m1 = self._count_m1.to(device)
        self._count_m2 = self._count_m2.to(device)
        if self._svd_cache is not None:
            self._svd_cache = (
                self._svd_cache[0].to(device),
                self._svd_cache[1].to(device),
            )
        return self

    def state_dict(self) -> dict:
        """Get a dict representation of the object for serialization"""
        return {
            "dim": self.dim,
            "moment1": self.moment1,
            "moment2": self.moment2,
            "center": self._center,
            "count_m1": self._count_m1,
            "count_m2": self._count_m2,
        }

    @staticmethod
    def load_state_dict(state: dict) -> "PrincipalComponents":
        """Create a new PrincipalComponents object from a state dict."""
        obj = PrincipalComponents(
            dim=state["dim"],
            center=state["center"],
            device=state["moment1"].device,
            dtype=state["moment1"].dtype,
        )
        obj.moment1 = state["moment1"].clone()
        obj.moment2 = state["moment2"].clone()
        obj._count_m1 = state["count_m1"].clone()
        obj._count_m2 = state["count_m2"].clone()
        return obj

    def add_batch_vectors(self, vectors) -> Self:
        # Update running estimate of moment1
        is_valid = ~torch.isnan(vectors)
        total_m1 = self.moment1 * self._count_m1 + torch.nansum(vectors, dim=0)
        self._count_m1 = self._count_m1 + torch.sum(is_valid, dim=0)
        self.moment1 = torch.where(self._count_m1 > 0, total_m1 / self._count_m1, 0)

        # Update running estimate of moment2
        is_valid2 = is_valid[:, None, :] * is_valid[:, :, None]
        total_m2 = self.moment2 * self._count_m2 + torch.nansum(
            vectors[:, :, None] * vectors[:, None, :], dim=0
        )
        self._count_m2 = self._count_m2 + torch.sum(is_valid2, dim=0)
        self.moment2 = torch.where(self._count_m2 > 0, total_m2 / self._count_m2, 0)

        # Invalidate cache
        self._svd_cache = None

        return self

    @property
    def mean(self) -> torch.Tensor:
        if self._center:
            return self.moment1
        else:
            return torch.zeros_like(self.moment1)

    @property
    def cov(self):
        return self.moment2 - self.mean[:, None] * self.mean[None, :]

    def _check_no_missing_moments(self, enforce_nonzero_only: bool = False):
        # Start with the strict test that there is >0 data for every feature and every pair of
        # features.
        if torch.all(self._count_m1 == 0):
            raise ValueError("No data has been added to this object.")
        elif torch.any(self._count_m1 == 0):
            idx = torch.where(self._count_m1 == 0)
            raise ValueError(f"Some features never had data: {idx}")
        elif torch.any(self._count_m2 == 0):
            ij = torch.where(torch.tril(self._count_m2 == 0))
            raise ValueError(f"Some pairs of features never had data: {ij}")

        if enforce_nonzero_only:
            return

        # Move on to the 'sensible' test that there are at least as many observations as features.
        if torch.any(self._count_m1 < self.dim):
            idx = torch.where(self._count_m1 < self.dim)
            raise ValueError(f"Some features have less data than there are dimensions: {idx}")
        elif torch.any(self._count_m2 < self.dim):
            ij = torch.where(torch.tril(self._count_m2 < self.dim))
            raise ValueError(f"Some pairs have less data than there are dimensions: {ij}")

    def interpolate(
        self, vecs: torch.Tensor, method: Literal["mean", "zero", "sample"] = "mean"
    ) -> torch.Tensor:
        """Fill in missing values in vecs using conditional Gaussian.

        Args:
            vecs: torch.Tensor of shape (n_samples, dim)
            method: str, one of ["mean", "zero", "sample"]
                - "mean": fill in missing values with the conditional mean computed from the
                    observed values
                - "zero": fill in missing values with zeros
                - "sample": fill in missing values with samples conditional on the observed values

        Returns:
            torch.Tensor of shape (n_samples, dim) with missing values filled in.\
        """
        self._check_no_missing_moments()
        nan_mask = torch.isnan(vecs)
        if method.lower() == "zero":
            return torch.where(nan_mask, torch.zeros_like(vecs), vecs)
        elif method.lower() in ["mean", "sample"]:
            mu, cov = self.mean, self.cov

            filled_vecs = vecs.clone()
            for i in range(vecs.shape[0]):
                missing = nan_mask[i]
                observed = ~missing
                if torch.any(missing):
                    mu_A = mu[missing]
                    mu_B = mu[observed]
                    cov_AA = cov[missing][:, missing]
                    cov_AB = cov[missing][:, observed]
                    cov_BB = cov[observed][:, observed]

                    x_B = vecs[i, observed]
                    mu_cond = mu_A + cov_AB @ torch.linalg.solve(cov_BB, (x_B - mu_B))

                    if method.lower() == "mean":
                        filled_vecs[i, missing] = mu_cond
                    elif method.lower() == "sample":
                        cov_cond = cov_AA - cov_AB @ torch.linalg.solve(cov_BB, cov_AB.T)
                        filled_vecs[i, missing] = mu_cond + torch.cholesky(
                            cov_cond
                        ) @ torch.randn_like(mu_cond)
            return filled_vecs
        else:
            assert_never(method)

    def whiten(self, vecs: torch.Tensor, zca: bool = False) -> torch.Tensor:
        """Whiten vecs using the covariance matrix of this object."""
        if torch.any(torch.isnan(vecs)):
            raise ValueError(
                "Cannot call whiten() with NaN values in the input."
                "Suggestion: first call interpolate() to fill in missing values."
            )
        self._check_no_missing_moments()
        u, s = self.spectral_decomposition()
        z = (vecs - self.mean[None, :]) @ u @ torch.diag(1 / torch.sqrt(s))
        if zca:
            return z @ u.T
        else:
            return z

    def suggest_dimensions(self, frac_variance: float) -> int:
        """Suggest the number of dimensions to retain to explain a given fraction of variance."""
        if not (0 < frac_variance <= 1):
            raise ValueError("percent_variance must be between 0.0 and 1.0")
        _, eigs = self.spectral_decomposition()
        total_variance = torch.sum(eigs)
        cumulative_variance = torch.cumsum(eigs, dim=0)
        num_dims = torch.searchsorted(cumulative_variance, frac_variance * total_variance)
        return num_dims.item() + 1  # Convert to 1-based indexing

    def reconstruct(self, low_d_vecs: torch.Tensor) -> torch.Tensor:
        """Reconstruct high-dimensional data from its low-dimensional representation."""
        _, k = low_d_vecs.shape
        if k > self.dim:
            raise ValueError("k cannot be greater than the original dimensionality.")
        u, _ = self.spectral_decomposition()
        return low_d_vecs @ u[:, :k].T + self.mean[None, :]

    def reduce_dim(self, vecs: torch.Tensor, k: int, original_space: bool = False):
        """Reduce the dimensionality of vecs to k dimensions using the first k principal components
        of this object.

        Args:
            vecs: torch.Tensor of shape (n_samples, dim), data to reduce
            k: int, number of top dimensions to keep
            original_space: bool, default False. If False, return (n_samples, k) projection of vecs.
                If True, project back up to the original space and return (n_samples, dim) tensor;
                equivalent to subtracting off the least significant dim-k parts of the signal.
        """
        if torch.any(torch.isnan(vecs)):
            raise ValueError(
                "Cannot call reduce_dim() with NaN values in the input."
                "Suggestion: first call interpolate() to fill in missing values."
            )
        self._check_no_missing_moments()
        u, _ = self.spectral_decomposition()
        projection = (vecs - self.mean[None, :]) @ u[:, :k]
        if original_space:
            return projection @ u[:, :k].T + self.mean[None, :]
        else:
            return projection

    def spectral_decomposition(self):
        """Returns (eigenvecs, eigenvals) of the covariance matrix, using previously cached result
        if the moments haven't since been updated.
        """
        self._check_no_missing_moments(enforce_nonzero_only=True)
        if self._svd_cache is None:
            self._svd_cache = torch.linalg.svd(self.cov)
        return self._svd_cache[:2]

    def _normalized_cov(self) -> torch.Tensor:
        c = self.cov
        return c / torch.sqrt(torch.sum(c**2))

    def subspace_similarity(self, other: "PrincipalComponents"):
        """Calculate similarity of this subspace and another subspace using the frobenius
        inner-product of their covariance matrices. This is equivalent to summing over PCs of
        either subspace and getting the variance of the projection of the other data onto that
        PC, scaled by the variance in this space.

        It's also convenient that the square of this similarity measure multiplied by the ambient
        dimension is equivalent to the n2 measure of effective dimensionality, i.e.
        effective_dim = dim * similarity**2.
        """
        if self._center != other._center:
            warnings.warn("Subspaces have different centering; results may not be meaningful.")
        self._check_no_missing_moments(enforce_nonzero_only=True)

        mat_a, mat_b = self._normalized_cov(), other._normalized_cov()

        return torch.clip(torch.sum(mat_a * mat_b), -1, 1)

    def subspace_similarity_null(
        self, other: "PrincipalComponents", n_samples: int = 1000, progbar: bool = False
    ):
        if self._center != other._center:
            warnings.warn("Subspaces have different centering; results may not be meaningful.")
        self._check_no_missing_moments(enforce_nonzero_only=True)

        mat_a, mat_b = self._normalized_cov(), other._normalized_cov()

        # When randomly rotating the matrices, only the eigenvalues end up mattering. This lets
        # us replace the sum of matrix products with a sum of matrix-vector products, which is
        # faster.
        eigs_a, eigs_b = torch.linalg.eigvalsh(mat_a), torch.linalg.eigvalsh(mat_b)

        def sample_sim(e_a, e_b):
            q, _ = torch.linalg.qr(
                torch.randn(self.dim, self.dim, device=mat_a.device, dtype=mat_a.dtype)
            )
            # We need to calculate trace(diag(e_a) @ q @ diag(e_b) @ q.T), but more
            # efficiently. Left-multiplying by a diagonal matrix is equivalent to scaling the
            # rows of the matrix, so diag(e_a) @ q is more efficiently done as (e_a[:,
            # None]*q) and (diag(e_b) @ q.T).T is (e_b[None,:]*q). Then, use the fact that
            # trace(A @ B) = sum(A.T * B), so we get the following quick calculation:
            return torch.sum((e_a[:, None] * q) * (e_b[None, :] * q))

        iterator = trange(n_samples, desc="Sampling null", disable=not progbar, leave=False)
        return [sample_sim(eigs_a, eigs_b) for _ in iterator]

    def subspace_similarity_null_normal(self, other: "PrincipalComponents"):
        if self._center != other._center:
            warnings.warn("Subspaces have different centering; results may not be meaningful.")
        self._check_no_missing_moments(enforce_nonzero_only=True)

        mat_a, mat_b = self._normalized_cov(), other._normalized_cov()

        eigs_a, eigs_b = torch.linalg.eigvalsh(mat_a), torch.linalg.eigvalsh(mat_b)

        d = self.dim
        trace_a, trace_b = torch.sum(eigs_a), torch.sum(eigs_b)
        sq_trace_a, sq_trace_b = trace_a**2, trace_b**2
        trace_sq_a, trace_sq_b = torch.sum(eigs_a**2), torch.sum(eigs_b**2)
        mean = trace_a * trace_b / d
        moment2 = (
            (d + 1) * sq_trace_a * sq_trace_b
            - 2 * sq_trace_a * trace_sq_b
            - 2 * trace_sq_a * sq_trace_b
        ) / (d * (d - 1) * (d + 2)) + 2 * trace_sq_a * trace_sq_b / ((d - 1) * (d + 2))
        variance = moment2 - mean**2
        stdev = torch.sqrt(variance)
        return mean.item(), stdev.item()

    def effective_dim(self, method="n2"):
        """Calculate a scale-invariant measure of the total dimensionality of the data. If the
        covariance is isotropic, and we have plenty of data, this is equivalent to the data
        dimension.

        Del Giudice, Marco. “Effective Dimensionality: A Tutorial.” Multivariate Behavioral
            Research 56 (March 7, 2021): 527–42. </div>
        """
        _, eigs = self.spectral_decomposition()
        eigs_normed = eigs / eigs.sum()
        match method:
            case "n1":
                return torch.prod(eigs_normed ** (-eigs_normed))
            case "n2":
                return torch.sum(eigs_normed) ** 2 / torch.sum(eigs_normed**2)
            case "nInf":
                return torch.sum(eigs_normed) / torch.max(eigs_normed)
            case "nC":
                return self.dim - self.dim**2 / torch.sum(eigs) ** 2 * torch.var(eigs)
            case _:
                assert_never(method)
