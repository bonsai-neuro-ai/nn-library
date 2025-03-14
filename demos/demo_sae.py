import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
from tqdm.auto import trange

from nn_lib.models import SparseAutoEncoder

in_dim = 3
num_tokens = 30

sae = SparseAutoEncoder(input_dim=in_dim, hidden_dim=num_tokens, beta_l1=0.1)

init_params = {k: v.clone() for k, v in sae.named_parameters()}

# %% Create a data generator by sampling a mixture of Gaussians.

cat = Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
means = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
covs = torch.eye(3).repeat(3, 1, 1) / 50
mixture = MixtureSameFamily(cat, MultivariateNormal(means, covs))

x = mixture.sample((1000,))

# %% Train the VQ-VAE.

optimizer, scheduler = sae.configure_optimizers()
history = []
for step in trange(5000):
    optimizer.zero_grad()
    info = sae.training_step(x, step)
    history.append({"step": step, **{k: v.item() for k, v in info.items()}})
    info["loss"].backward()
    optimizer.step()
    # scheduler.step()

df = pd.DataFrame(history)
sns.lineplot(data=df, x="step", y="reconstruction_loss", label="reconstruction")
sns.lineplot(data=df, x="step", y="sparsity_loss", label="sparsity")
plt.ylabel("loss")
plt.show()

# %%

sae.eval()
with torch.no_grad():
    x_hat, z = sae(x)
    nnz = torch.sum(z.abs() > 1e-6, dim=1)

plt.figure()
plt.hist(nnz, bins=np.arange(20))
plt.xticks(np.arange(20) + 0.5, np.arange(20))
plt.xlabel("Number of non-zero activations")
plt.ylabel("Number of samples")
plt.show()

z_nonzero = torch.where(z > 1e-6, z, torch.nan)
z_means = torch.nanmean(z_nonzero, dim=0)
useful_tokens = z_means > 1e-3
vectors = sae.codebook.detach()[:, useful_tokens] * z_means[useful_tokens]

plt.figure()
ax = plt.subplot(111, projection="3d")
ax.scatter(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], c="r", marker="o", s=5)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c="k", marker=".", s=5)
for xyz in vectors.T:
    ax.quiver(0, 0, 0, *xyz, color="b")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.axis("equal")
plt.show()
