import time
from collections import defaultdict
from typing import assert_never

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn import functional as F

from nn_lib.utils import xval_nuc_norm_cross_cov


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

    @property
    def cov_x(self):
        return torch.eye(self.n) + self.proj_x.T @ self.proj_x

    @property
    def cov_y(self):
        return torch.eye(self.n) + self.proj_y.T @ self.proj_y

    @property
    def cov_xy(self):
        return self.proj_x.T @ self.proj_y

    def sample(self, m):
        z = torch.randn(m, self.k)
        x = torch.randn(m, self.n) + z @ self.proj_x
        y = torch.randn(m, self.n) + z @ self.proj_y
        return x, y


class ConvDataGenerator:
    """Like DataGenerator but where it simulates convolutional feature maps with local correlations"""

    def __init__(self, h, w, n, k, kernel_size=3):
        self.h = h
        self.w = w
        self.n = n
        self.k = k
        self.kernel_size = kernel_size
        in_dim = k * kernel_size * kernel_size
        self.proj_x = torch.randn(n, k, kernel_size, kernel_size) / np.sqrt(in_dim)
        self.proj_y = torch.randn(n, k, kernel_size, kernel_size) / np.sqrt(in_dim)

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

        x = torch.zeros(1, C, H, W)
        y = F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)

        M = y.numel()
        N = C * H * W

        B = torch.zeros(M, N)

        for i in range(N):
            basis = torch.zeros_like(x)
            basis.view(-1)[i] = 1.0

            out = F.conv2d(basis, w, stride=stride, padding=padding, dilation=dilation)
            B[:, i] = out.view(-1)

        return B

    @staticmethod
    def conv_operator(weight, out_h, out_w):
        """
        weight: (c, kc, kh, kw)
        Returns: C of shape (c*h*w, kc*(h+kh-1)*(w+kw-1))
        """
        c, kc, kh, kw = weight.shape
        in_h = out_h + kh - 1
        in_w = out_w + kw - 1
        ww = F.fold(
            weight.reshape(c, kc * kh * kw, 1).repeat(1, 1, out_h * out_w), (in_h, in_w), (kh, kw)
        )
        pass

    def sample(self, m):
        z = torch.randn(m, self.k, self.h + self.kernel_size - 1, self.w + self.kernel_size - 1)
        x = torch.randn(m, self.n, self.h, self.w) + F.conv2d(z, self.proj_x)
        y = torch.randn(m, self.n, self.h, self.w) + F.conv2d(z, self.proj_y)
        return x, y


def nuc_norm_plugin(x, y):
    m = x.shape[0]
    cross_cov = x.T @ y / m
    return torch.linalg.norm(cross_cov, ord="nuc")


def run_nuc_norm_test(gen: DataGenerator | ConvDataGenerator, method, m):
    if method == "true":
        norm = torch.linalg.norm(gen.cov_xy, ord="nuc")
        elapsed = 0.0
    else:
        x, y = gen.sample(m)
        x, y = x.flatten(start_dim=1).cuda(), y.flatten(start_dim=1).cuda()
        tstart = time.time()
        match method:
            case "plugin":
                norm = nuc_norm_plugin(x, y)
            case "LOO[ab]":
                norm = xval_nuc_norm_cross_cov(x, y, method="ab")
            case "LOO[ortho]":
                norm = xval_nuc_norm_cross_cov(x, y, method="orthogonalize")
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

# Generate data with some shared low-rank correlation
norms = defaultdict(list)
n_inner = 4
gen = DataGenerator(n=1000, k=100)
# gen = ConvDataGenerator(n=64, k=64, h=32, w=32, kernel_size=3)
methods = ["true", "plugin", "LOO[ab]", "LOO[ortho]"]
results = []
for m in np.logspace(1, 4, 7).astype(int):
    print("m =", m)
    for method in methods:
        for j in range(n_inner):
            results.append(run_nuc_norm_test(gen, method, m=m))
            if method == "true":
                break
df = pd.DataFrame(results)

# %%

true_val = torch.linalg.norm(gen.cov_xy, ord="nuc")
sns.lineplot(data=df, x="m", y="norm", hue="method")
plt.xscale("log")
plt.ylim(0.5 * true_val, 1.5 * true_val)
plt.savefig(f"nuc_norm_bias_n{gen.n}_k{gen.k}.png")
plt.show()


# %%

sns.lineplot(data=df, x="m", y="time", hue="method")
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"nuc_norm_timing_n{gen.n}_k{gen.k}.png")
plt.show()
