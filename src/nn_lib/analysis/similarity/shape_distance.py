import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from .utils import assert_repeatable_iter_factory, BatchIteratorFactory
from nn_lib.utils import (
    xval_nuc_norm_cross_cov,
    RunningAverage,
    calculate_moments_batchwise,
    moments_to_covs,
)


def distance(
    trace_cov_xx: torch.Tensor, trace_cov_yy: torch.Tensor, nuc_norm_xy: torch.Tensor, scaled: bool
) -> torch.Tensor:
    if scaled:
        # Riemannian Shape Distance (arc length):
        cosine_similarity = nuc_norm_xy / torch.sqrt(trace_cov_xx * trace_cov_yy)
        return torch.arccos(torch.clip(cosine_similarity, -1.0, 1.0))
    else:
        # Procrustes size-and-shape distance (Euclidean):
        return torch.sqrt(torch.clip(trace_cov_xx + trace_cov_yy - 2 * nuc_norm_xy, 0.0, None))


class ShapeDistance(StreamingComparator):
    """Computes the (Procrustes) Shape Distance between neural representations X and Y.

    Args:
        - centered: if True, centers data like X-mean(X,dim=0) and compares covariance and
            cross-covariance matrices. If False, compares uncentered second moments.
        - scaled: if True, scales the data like X/norm(X, ord="fro") and all distances are measured
            as arc-lengths (radians); this is 'Riemannian shape distance' in the literature. if
            False, no scaling is applied and a Euclidean distance is measured.
    """

    def __init__(self, centered: bool, scaled: bool):
        self.centered = centered
        self.scaled = scaled

    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory) -> torch.Tensor:
        moments = calculate_moments_batchwise(batch_iterator_factory())
        covs = moments_to_covs(moments, self.centered)
        return distance(
            torch.trace(covs["cov_0_0"]),
            torch.trace(covs["cov_1_1"]),
            torch.linalg.norm(covs["cov_0_1"], ord="nuc"),
            scaled=self.scaled,
        )


class CrossValidatedShapeDistance(StreamingComparator):
    def __init__(self, centered: bool, scaled: bool):
        self.centered = centered
        self.scaled = scaled

    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory):
        # We will re-use the iterator, so first step is to assert that it is repeatable
        assert_repeatable_iter_factory(batch_iterator_factory)

        # First-pass: calculate moments and get low-bias estimate of the 'xx' and 'yy' terms
        moments = calculate_moments_batchwise(batch_iterator_factory())
        covs = moments_to_covs(moments, self.centered)
        m = moments["moment1_0"].count

        # Precompute SVD of xy; this 'global' SVD is then passed into the xval_nuc_norm_cross_cov
        # function which will calculate 'updated' SVDs.
        svd = torch.linalg.svd(covs["cov_0_1"], full_matrices=True)

        # Second-pass: call xval_nuc_norm_cross_cov per batch, passing in svd for 'global' stats
        xval_nuc_norm_xy = RunningAverage()
        for batch_x, batch_y in batch_iterator_factory():
            if self.centered:
                batch_x = batch_x - moments["moment1_0"].avg.unsqueeze(0)
                batch_y = batch_y - moments["moment1_1"].avg.unsqueeze(0)

            batch_avg_nuc_norm = xval_nuc_norm_cross_cov(
                batch_x, batch_y, svd_cross_cov=svd, m_total=m, method="ab"
            )
            xval_nuc_norm_xy.update(batch_avg_nuc_norm, batch_count=batch_x.shape[0])

        return distance(
            torch.trace(covs["cov_0_0"]),
            torch.trace(covs["cov_1_1"]),
            xval_nuc_norm_xy.avg,
            scaled=self.scaled,
        )
