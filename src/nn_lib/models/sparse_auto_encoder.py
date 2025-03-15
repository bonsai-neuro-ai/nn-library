from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn


class SparseAutoEncoder(LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, beta_l1: float = 0.01):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_out_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.beta_l1 = beta_l1

    @property
    def codebook(self) -> torch.Tensor:
        return self.decoder.weight + self.in_out_bias[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.in_out_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.in_out_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, lr_scheduler

    def training_step(self, x: torch.Tensor, batch_idx: int):
        x_hat, z = self(x)
        sparsity_loss = torch.mean(torch.sum(z.abs(), dim=1))
        reconstruction_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
        return {
            "loss": reconstruction_loss + self.beta_l1 * sparsity_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
        }

    def validation_step(self, *args: Any, **kwargs: Any):
        return self.training_step(*args, **kwargs)


__all__ = [
    "SparseAutoEncoder",
]
