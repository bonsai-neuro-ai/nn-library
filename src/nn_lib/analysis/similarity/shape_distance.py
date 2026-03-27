from collections import defaultdict

import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from .utils import assert_repeatable_iter_factory, BatchIteratorFactory, RunningAverage
from nn_lib.utils import xval_nuc_norm_cross_cov


def _calculate_moments(batch_iterator_factory: BatchIteratorFactory):
    moments = defaultdict(RunningAverage)

    for x, y in batch_iterator_factory():
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        m, n_x = x.shape
        _, n_y = y.shape

        moments["moment1_x"].update(torch.mean(x, dim=0), m)
        moments["moment1_y"].update(torch.mean(y, dim=0), m)
        moments["moment2_xx"].update(torch.einsum("mi,mj->ij", x, x) / m, m)
        moments["moment2_yy"].update(torch.einsum("mi,mj->ij", y, y) / m, m)
        moments["moment2_xy"].update(torch.einsum("mi,mj->ij", x, y) / m, m)

    return moments


def _moments_to_covs(
    moments: dict[str, RunningAverage], centered: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m1x = moments["moment1_x"].avg
    m1y = moments["moment1_y"].avg
    m2xx = moments["moment2_xx"].avg
    m2yy = moments["moment2_yy"].avg
    m2xy = moments["moment2_xy"].avg

    if centered:
        cov_xx = m2xx - m1x[:, None] * m1x[None, :]
        cov_yy = m2yy - m1y[:, None] * m1y[None, :]
        cov_xy = m2xy - m1x[:, None] * m1y[None, :]
    else:
        cov_xx = m2xx
        cov_yy = m2yy
        cov_xy = m2xy

    return cov_xx, cov_yy, cov_xy


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
        moments = _calculate_moments(batch_iterator_factory)
        cov_xx, cov_yy, cov_xy = _moments_to_covs(moments, self.centered)
        return distance(
            torch.trace(cov_xx),
            torch.trace(cov_yy),
            torch.linalg.norm(cov_xy, ord="nuc"),
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
        moments = _calculate_moments(batch_iterator_factory)
        cov_xx, cov_yy, cov_xy = _moments_to_covs(moments, self.centered)
        m = moments["moment1_x"].count

        # Precompute SVD of xy; this 'global' SVD is then passed into the xval_nuc_norm_cross_cov
        # function which will calculate 'updated' SVDs.
        svd = torch.linalg.svd(cov_xy, full_matrices=True)

        # Second-pass: call xval_nuc_norm_cross_cov per batch, passing in svd for 'global' stats
        xval_nuc_norm_xy = RunningAverage()
        for batch_x, batch_y in batch_iterator_factory():
            if self.centered:
                batch_x = batch_x - moments["moment1_x"].avg.unsqueeze(0)
                batch_y = batch_y - moments["moment1_y"].avg.unsqueeze(0)

            batch_avg_nuc_norm = xval_nuc_norm_cross_cov(
                batch_x, batch_y, svd_cross_cov=svd, m_total=m, method="ab"
            )
            xval_nuc_norm_xy.update(batch_avg_nuc_norm, batch_count=batch_x.shape[0])

        return distance(
            torch.trace(cov_xx),
            torch.trace(cov_yy),
            xval_nuc_norm_xy.avg,
            scaled=self.scaled,
        )
