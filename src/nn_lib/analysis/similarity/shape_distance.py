from collections import defaultdict

import torch

from nn_lib.analysis.similarity.comparator import StreamingComparator
from .utils import assert_repeatable_iter_factory, BatchIteratorFactory, RunningAverage


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

    def distance(
        self, cov_xx: torch.Tensor, cov_yy: torch.Tensor, cov_xy: torch.Tensor
    ) -> torch.Tensor:
        norm_xx = torch.trace(cov_xx)
        norm_yy = torch.trace(cov_yy)
        norm_xy = torch.linalg.norm(cov_xy, ord="nuc")
        if self.scaled:
            # Riemannian Shape Distance (arc length):
            cosine_similarity = norm_xy / torch.sqrt(norm_xx * norm_yy)
            return torch.arccos(torch.clip(cosine_similarity, -1.0, 1.0))
        else:
            # Procrustes size-and-shape distance (Euclidean):
            return torch.sqrt(torch.clip(norm_xx + norm_yy - 2 * norm_xy, 0.0, None))

    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory) -> torch.Tensor:
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

        m1x = moments["moment1_x"].avg
        m1y = moments["moment1_y"].avg
        m2xx = moments["moment2_xx"].avg
        m2yy = moments["moment2_yy"].avg
        m2xy = moments["moment2_xy"].avg

        if self.centered:
            cov_xx = m2xx - m1x[:, None] * m1x[None, :]
            cov_yy = m2yy - m1y[:, None] * m1y[None, :]
            cov_xy = m2xy - m1x[:, None] * m1y[None, :]
        else:
            cov_xx = m2xx
            cov_yy = m2yy
            cov_xy = m2xy

        return self.distance(cov_xx, cov_yy, cov_xy)
