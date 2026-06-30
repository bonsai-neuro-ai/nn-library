from collections import defaultdict
from typing import Iterable

import torch


class RunningAverage[T]:
    """
    Incrementally tracks a weighted running mean of values seen in batches, without needing to
    keep all the data in memory. `T` is typically a `torch.Tensor` but can be anything supporting
    `+`, `-`, and scalar multiplication/division (e.g. a plain float).
    """

    def __init__(self):
        self.avg: T | None = None
        self.count: int = 0

    def update(self, batch_avg: T, batch_count: int):
        """
        Fold in a new batch's average.

        :param batch_avg: the mean value of the new batch (same type/shape as previous batches).
        :param batch_count: number of samples the new batch's average was computed over, used to
            weight its contribution to the running mean.
        """
        if self.avg is None:
            self.avg = batch_avg
        else:
            self.avg = self.avg + (batch_avg - self.avg) * batch_count / (self.count + batch_count)
        self.count += batch_count


def calculate_moments_batchwise[T](
    batches: Iterable[tuple[T, ...]],
) -> dict[str, RunningAverage[T]]:
    """
    Stream over batches of one or more aligned tensors and accumulate first and second moments
    (means and cross-covariance-like products) for each, without materializing the full dataset.

    Each item yielded by `batches` is a tuple of tensors of shape (batch, ...) representing
    aligned variables (e.g. activations from different layers/models on the same inputs). Each
    tensor is flattened to (batch, features) before accumulating.

    :param batches: iterable of tuples of same-length tensors, one tuple per minibatch.
    :return: dict mapping "moment1_{i}" -> running mean of variable i, and "moment2_{i}_{j}"
        (for i <= j) -> running mean of x_i^T x_j / batch_size, the uncentered second moment.
        Pass this dict to `moments_to_covs` to get (co)variance matrices.
    """
    moments = defaultdict(RunningAverage)

    for batch in batches:
        for i, x in enumerate(batch):
            x = x.flatten(start_dim=1)
            m, n_x = x.shape
            moments[f"moment1_{i}"].update(torch.mean(x, dim=0), m)
            for j, y in enumerate(batch):
                if i > j:
                    continue
                moments[f"moment2_{i}_{j}"].update(torch.einsum("mi,mj->ij", x, y) / m, m)

    return dict(moments)


def moments_to_covs[T](moments: dict[str, RunningAverage[T]], centered: bool) -> dict[str, T]:
    """
    Convert the moments dict produced by `calculate_moments_batchwise` into (co)variance matrices.

    :param moments: output of `calculate_moments_batchwise`.
    :param centered: if True, subtract the outer product of means to get true covariance matrices
        (i.e. center the data); if False, return the raw uncentered second moments.
    :return: dict mapping "cov_{i}_{j}" -> covariance (or uncentered second moment) matrix between
        variables i and j.
    """
    out = {}
    for k, v in moments.items():
        if k.startswith("moment2"):
            _, i, j = k.split("_")
            i, j = int(i), int(j)
            if centered:
                moment1_i = moments[f"moment1_{i}"].avg
                moment1_j = moments[f"moment1_{j}"].avg
                out[f"cov_{i}_{j}"] = v.avg - moment1_i[:, None] * moment1_j[None, :]
            else:
                out[f"cov_{i}_{j}"] = v.avg

    return out


__all__ = [
    "RunningAverage",
    "calculate_moments_batchwise",
    "moments_to_covs",
]
