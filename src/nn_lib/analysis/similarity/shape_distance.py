from typing import Literal

import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from nn_lib.utils import (
    xval_nuc_norm_cross_cov,
    RunningAverage,
    calculate_moments_batchwise,
)
from .utils import assert_repeatable_iter_factory, BatchIteratorFactory


def distance(
    trace_cov_xx: torch.Tensor, trace_cov_yy: torch.Tensor, nuc_norm_xy: torch.Tensor, scaled: bool
) -> torch.Tensor:
    """Compute the shape distance given precomputed (co)variance summaries.

    :param trace_cov_xx: trace of the covariance (or uncentered second moment) of X.
    :param trace_cov_yy: trace of the covariance (or uncentered second moment) of Y.
    :param nuc_norm_xy: nuclear norm of the cross-covariance between X and Y.
    :param scaled: if True, return the Riemannian arc-length (arccos of cosine similarity);
        if False, return the Euclidean Procrustes size-and-shape distance.
    """
    if scaled:
        # Riemannian Shape Distance (arc length):
        cosine_similarity = nuc_norm_xy / torch.sqrt(trace_cov_xx * trace_cov_yy)
        return torch.arccos(torch.clip(cosine_similarity, -1.0, 1.0))
    else:
        # Procrustes size-and-shape distance (Euclidean):
        return torch.sqrt(torch.clip(trace_cov_xx + trace_cov_yy - 2 * nuc_norm_xy, 0.0, None))


def select_moments(
    moments: dict[str, RunningAverage], centered: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centered:
        return (
            moments["cov_0_0"].avg,
            moments["cov_1_1"].avg,
            moments["cov_0_1"].avg,
        )
    else:
        return (
            moments["moment2_0_0"].avg,
            moments["moment2_1_1"].avg,
            moments["moment2_0_1"].avg,
        )


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
        moments = calculate_moments_batchwise(batch_iterator_factory(), covariances=True)
        m00, m11, m01 = select_moments(moments, self.centered)
        return distance(
            torch.trace(m00),
            torch.trace(m11),
            torch.linalg.norm(m01, ord="nuc"),
            scaled=self.scaled,
        )


class CrossValidatedShapeDistance(StreamingComparator):
    """
    Cross-validated variant of `ShapeDistance` that corrects for the upward bias in the nuclear
    norm of the cross-covariance matrix caused by finite sample size (analogous to how
    cross-validation corrects for overfitting). Requires two passes over the data so the
    `batch_iterator_factory` must be repeatable (see `assert_repeatable_iter_factory`).

    :param centered: if True, center representations before computing (co)variances.
    :param scaled: if True, return Riemannian arc-length distance; if False, Euclidean Procrustes.
    :param xval_method: algorithm for leave-one-batch-out nuclear norm estimation. Options are
        "brute_force", "rank1", "orthogonalize", and "ab" (see `xval_nuc_norm_cross_cov`).
    """

    def __init__(
        self,
        centered: bool,
        scaled: bool,
        xval_method: Literal["brute_force", "rank1", "orthogonalize", "ab"] = "orthogonalize",
    ):
        self.centered = centered
        self.scaled = scaled
        self.method = xval_method

    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory):
        # We will re-use the iterator, so first step is to assert that it is repeatable
        assert_repeatable_iter_factory(batch_iterator_factory)

        # First-pass: calculate moments and get low-bias estimate of the 'xx' and 'yy' terms
        moments = calculate_moments_batchwise(batch_iterator_factory(), covariances=True)
        m = moments["moment1_0"].count

        # Degrees of freedom used to normalize the cross-covariance. When 'centered' is True, the
        # empirical mean is used and all cov estimators divide the sum of squares by (m-1). When
        # centered is False, we assume mu=zeros for x and y and covs are normalized by just m.
        dof = 1 if self.centered else 0

        # Pick the correctly-normalized cross-cov: cov_0_1 (centered, 1/(m-1)) or moment2_0_1
        # (uncentered, 1/m).
        m00, m11, m01 = select_moments(moments, self.centered)

        # Precompute SVD of the cross-cov; this 'global' SVD is passed into xval_nuc_norm_cross_cov
        # which computes 'downdated' (leave-one-out) SVDs.
        svd = torch.linalg.svd(m01, full_matrices=False)

        # Second-pass: call xval_nuc_norm_cross_cov per batch, passing in svd for 'global' stats
        xval_nuc_norm_xy = RunningAverage()
        for batch_x, batch_y in batch_iterator_factory():
            if self.centered:
                # TODO - cross-validate the means too
                batch_x = batch_x - moments["moment1_0"].avg.unsqueeze(0)
                batch_y = batch_y - moments["moment1_1"].avg.unsqueeze(0)

            # DOF note: the per-sample downdates inside xval_nuc_norm methods subtract x_i y_i /
            # m_total, so m_total must match the normalization baked into m01 (i.e. 1/(m - dof)). In
            # other words, using m_total=m instead of m_total=m-dof results in small errors (failing
            # tests) when centered=True
            batch_avg_nuc_norm = xval_nuc_norm_cross_cov(
                batch_x, batch_y, svd_cross_cov=svd, m_total=m - dof, method=self.method
            )
            xval_nuc_norm_xy.update(batch_avg_nuc_norm, batch_count=batch_x.shape[0])

        return distance(
            torch.trace(m00),
            torch.trace(m11),
            xval_nuc_norm_xy.avg,
            scaled=self.scaled,
        )
