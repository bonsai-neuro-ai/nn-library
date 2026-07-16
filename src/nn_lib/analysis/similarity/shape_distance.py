from typing import Literal

import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from nn_lib.utils import (
    RunningCovariance,
    RunningAverage,
    XValStats,
    xval_nuc_norm_cross_cov,
    RunningVariance,
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


def _first_pass_moments(
    batch_iterator_factory: BatchIteratorFactory, centered: bool
) -> tuple[RunningVariance, RunningVariance, RunningCovariance]:
    stats_x = RunningVariance(centered=centered)
    stats_y = RunningVariance(centered=centered)
    stats_xy = RunningCovariance(centered=centered)

    for batch_x, batch_y in batch_iterator_factory():
        stats_x.update(batch_x)
        stats_y.update(batch_y)
        stats_xy.update(batch_x, batch_y)

    return stats_x, stats_y, stats_xy


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
        stats_x, stats_y, stats_xy = _first_pass_moments(batch_iterator_factory, self.centered)

        return distance(
            trace_cov_xx=torch.sum(stats_x.variance),
            trace_cov_yy=torch.sum(stats_y.variance),
            nuc_norm_xy=torch.linalg.norm(stats_xy.covariance, ord="nuc"),
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
        # Two passes over the data are needed, so the factory must be repeatable.
        assert_repeatable_iter_factory(batch_iterator_factory)

        # First pass: accumulate moments. These give the low-bias 'xx'/'yy' trace terms and the
        # cross-cov whose SVD drives the leave-one-out 'xy' term. select_moments picks the
        # correctly-normalized cross-cov: cov_0_1 (centered, 1/(m-1)) or moment2_0_1 (uncentered).
        stats_x, stats_y, stats_xy = _first_pass_moments(batch_iterator_factory, self.centered)

        # Package the global stats once. Supplying the means (only when centered) is what tells
        # the estimator to center each batch and use the (m-1) normalization; we always hand it
        # raw batches below, so centering happens in exactly one place and can't be applied twice.
        stats = XValStats.from_running_covariance(stats_xy)

        # Second pass: leave-one-out cross-cov nuclear norm, averaged over samples.
        xval_nuc_norm_xy = RunningAverage()
        for batch_x, batch_y in batch_iterator_factory():
            batch_avg_nuc_norm = xval_nuc_norm_cross_cov(
                batch_x, batch_y, method=self.method, stats=stats
            )
            xval_nuc_norm_xy.update(batch_avg_nuc_norm, batch_count=batch_x.shape[0])

        return distance(
            trace_cov_xx=torch.sum(stats_x.variance),
            trace_cov_yy=torch.sum(stats_y.variance),
            nuc_norm_xy=xval_nuc_norm_xy.avg,
            scaled=self.scaled,
        )
