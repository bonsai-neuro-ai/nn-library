from collections import defaultdict
from typing import Iterable

import torch


class RunningAverage[T]:
    def __init__(self):
        self.avg: T | None = None
        self.count: int = 0

    def update(self, batch_avg: T, batch_count: int):
        if self.avg is None:
            self.avg = batch_avg
        else:
            self.avg = self.avg + (batch_avg - self.avg) * batch_count / (self.count + batch_count)
        self.count += batch_count


def calculate_moments_batchwise[T](
    batches: Iterable[tuple[T, ...]],
) -> dict[str, RunningAverage[T]]:
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
