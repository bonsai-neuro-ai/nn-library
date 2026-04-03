import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import assert_never, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Memory
from torch.nn import functional as F

from nn_lib.analysis.similarity.utils import prep_conv_layers
from nn_lib.utils import xval_nuc_norm_cross_cov, RunningAverage
from nn_lib.utils.stats import calculate_moments_batchwise


def eye_like(A: torch.Tensor) -> torch.Tensor:
    return torch.eye(A.shape[0], device=A.device, dtype=A.dtype)


class DataGenerator:
    """Class to assist in generating gaussian-distributed data X and Y such that there is a
    true cov_x and cov_y and cov_xy we can calculate. This allows us to compare nuclear norms
    derived from various sample-based estimators to the true norm.
    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.proj_x = torch.randn(k, n) / np.sqrt(k)
        self.proj_y = torch.randn(k, n) / np.sqrt(k)
        self.cov_x = torch.eye(self.n) + self.proj_x.T @ self.proj_x
        self.cov_y = torch.eye(self.n) + self.proj_y.T @ self.proj_y
        self.cov_xy = self.proj_x.T @ self.proj_y

    def __str__(self):
        return f"DataGen_n{self.n}_k{self.k}"

    def __repr__(self):
        return self.__str__()

    def sample(self, m):
        z = torch.randn(m, self.k)
        x = torch.randn(m, self.n) + z @ self.proj_x
        y = torch.randn(m, self.n) + z @ self.proj_y
        return x, y


class ConvDataGenerator:
    """Like DataGenerator but where it simulates convolutional feature maps with local correlations"""

    def __init__(
        self, h, w, n, k, kernel_size=3, strategy="flatten", window: int = 1, device="cpu"
    ):
        self.h = h
        self.w = w
        self.n = n
        self.k = k
        self.kernel_size = kernel_size
        in_dim = k * kernel_size * kernel_size
        self.device = device
        self.proj_x = torch.randn(n, k, kernel_size, kernel_size, device=device) / np.sqrt(in_dim)
        self.proj_y = torch.randn(n, k, kernel_size, kernel_size, device=device) / np.sqrt(in_dim)
        self.vectorizer = partial(prep_conv_layers, conv_method=strategy, window_size=window)
        self.flat = "flatten" if strategy == "flatten" else f"window{window}"

    def __str__(self):
        return f"ConvDataGen_{self.flat}_h{self.h}_w{self.w}_n{self.n}_k{self.k}_kernel{self.kernel_size}"

    def __repr__(self):
        return self.__str__()

    @property
    def cov_x(self):
        proj_folded = self.fold_weights(
            self.proj_x, (self.k, self.h + self.kernel_size - 1, self.w + self.kernel_size - 1)
        )
        wwT = proj_folded @ proj_folded.T
        return wwT + eye_like(wwT)

    @property
    def cov_y(self):
        proj_folded = self.fold_weights(
            self.proj_y, (self.k, self.h + self.kernel_size - 1, self.w + self.kernel_size - 1)
        )
        wwT = proj_folded @ proj_folded.T
        return wwT + eye_like(wwT)

    @property
    def cov_xy(self):
        proj_folded_x = self.fold_weights(
            self.proj_x, (self.k, self.h + self.kernel_size - 1, self.w + self.kernel_size - 1)
        )
        proj_folded_y = self.fold_weights(
            self.proj_y, (self.k, self.h + self.kernel_size - 1, self.w + self.kernel_size - 1)
        )
        return proj_folded_x @ proj_folded_y.T

    @staticmethod
    def fold_weights(w, in_shape, stride=1, padding=0, dilation=1):
        C, H, W = in_shape
        out_channels, _, kh, kw = w.shape

        x = torch.zeros(1, C, H, W, device=w.device)
        y = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)

        M = y.numel()
        N = C * H * W

        B = torch.zeros(M, N, device=w.device)

        for i in range(N):
            basis = torch.zeros_like(x)
            basis.view(-1)[i] = 1.0

            out = F.conv2d(basis, w, stride=stride, padding=padding, dilation=dilation)
            B[:, i] = out.view(-1)

        return B

    def sample(self, m):
        z = torch.randn(
            m,
            self.k,
            self.h + self.kernel_size - 1,
            self.w + self.kernel_size - 1,
            device=self.device,
        )
        x = torch.randn(m, self.n, self.h, self.w, device=self.device) + F.conv2d(z, self.proj_x)
        y = torch.randn(m, self.n, self.h, self.w, device=self.device) + F.conv2d(z, self.proj_y)
        return self.vectorizer(x, y)


def batch_on_cuda(
    *tensors: torch.Tensor, batch_size: int
) -> Generator[tuple[torch.Tensor, ...], None, None]:
    i = 0
    while i < len(tensors[0]):
        batch = tuple(t[i : i + batch_size].cuda() for t in tensors)
        yield batch
        i += batch_size


def run_nuc_norm_test(gen: DataGenerator | ConvDataGenerator, method, m, uid):
    if method == "true":
        norm = torch.linalg.norm(gen.cov_xy, ord="nuc")
        elapsed = 0.0
    else:
        x, y = gen.sample(m)
        iter_factory = lambda: batch_on_cuda(
            x.flatten(start_dim=1), y.flatten(start_dim=1), batch_size=10
        )
        tstart = time.time()
        # first-pass: moment estimation
        moments = calculate_moments_batchwise(iter_factory())
        if method == "plugin":
            norm = torch.linalg.norm(moments["moment2_0_1"].avg, ord="nuc")
        else:
            svd = torch.linalg.svd(moments["moment2_0_1"].avg, full_matrices=False)
            norm = RunningAverage()
            for bx, by in iter_factory():
                match method:
                    case "LOO[ab]":
                        norm = xval_nuc_norm_cross_cov(
                            bx, by, method="ab", svd_cross_cov=svd, m_total=m
                        )
                    case "LOO[ortho]":
                        norm = xval_nuc_norm_cross_cov(
                            bx, by, method="orthogonalize", svd_cross_cov=svd, m_total=m
                        )
                    case _:
                        assert_never(method)
        elapsed = time.time() - tstart

    return {
        "norm": norm.item(),
        "method": method,
        "m": m,
        "n": gen.n,
        "rank": gen.k,
        "time": elapsed,
        "uid": uid,
    }


# %% Sanity-check: ConvDataGenerator correctly calculates covariance of noise fed through conv2d

gen = ConvDataGenerator(n=3, k=2, h=4, w=4)
x, y = gen.sample(10000)
x = x.flatten(start_dim=1)
y = y.flatten(start_dim=1)
fig, ax = plt.subplots(3, 3, figsize=(10, 10))

emp_cov_x = torch.einsum("mi,mj->ij", x, x) / (len(x) - 1)
ax[0, 0].imshow(gen.cov_x - eye_like(gen.cov_x), vmin=-1, vmax=1, cmap="icefire")
ax[0, 1].imshow(emp_cov_x - eye_like(emp_cov_x), vmin=-1, vmax=1, cmap="icefire")
ax[0, 2].imshow(torch.abs(emp_cov_x - gen.cov_x), vmin=0, vmax=1e-1, cmap="magma")

emp_cov_y = torch.einsum("mi,mj->ij", y, y) / (len(y) - 1)
ax[1, 0].imshow(gen.cov_y - eye_like(gen.cov_y), vmin=-1, vmax=1, cmap="icefire")
ax[1, 1].imshow(emp_cov_y - eye_like(emp_cov_y), vmin=-1, vmax=1, cmap="icefire")
ax[1, 2].imshow(torch.abs(emp_cov_y - gen.cov_y), vmin=0, vmax=1e-1, cmap="magma")

emp_ccov_xy = torch.einsum("mi,mj->ij", x, y) / (len(x) - 1)
ax[2, 0].imshow(gen.cov_xy, vmin=-0.5, vmax=0.5, cmap="icefire")
ax[2, 1].imshow(emp_ccov_xy, vmin=-0.5, vmax=0.5, cmap="icefire")
ax[2, 2].imshow(torch.abs(emp_ccov_xy - gen.cov_xy), vmin=0, vmax=1e-1, cmap="magma")

for a in ax.flatten():
    a.axis("off")

plt.show()

# %%

CONVOLUTIONAL = True

# Generate data with some shared low-rank correlation
norms = defaultdict(list)
n_inner = 4
gen = (
    ConvDataGenerator(
        n=64, k=64, h=16, w=16, kernel_size=3, strategy="window", window=1, device="cuda"
    )
    if CONVOLUTIONAL
    else DataGenerator(n=1000, k=100)
)
cache = Memory(Path(".cache") / str(gen))
run_nuc_norm_test = cache.cache(run_nuc_norm_test, ignore=["gen"])

methods = ["true", "plugin", "LOO[ab]", "LOO[ortho]"]
results = []
for m in np.logspace(1, 4, 7).astype(int):
    print("m =", m)
    for method in methods:
        for j in range(n_inner):
            results.append(run_nuc_norm_test(gen, method, m=m, uid=j))
            if method == "true":
                break
df = pd.DataFrame(results)

# %%

true_val = torch.linalg.norm(gen.cov_xy, ord="nuc")
sns.lineplot(data=df, x="m", y="norm", hue="method")
plt.xscale("log")
plt.ylim(0.5 * true_val, 1.5 * true_val)
plt.savefig(f"nuc_norm_bias_n{gen.n}_k{gen.k}_conv.png")
plt.show()


# %%

sns.lineplot(data=df, x="m", y="time", hue="method")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"nuc_norm_timing_n{gen.n}_k{gen.k}_conv.png")
plt.show()
