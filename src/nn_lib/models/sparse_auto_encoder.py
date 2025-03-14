from torch import nn
import torch
import torch.nn.functional as F


class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.in_out_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.in_out_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.in_out_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


__all__ = [
    "SparseAutoEncoder",
]
