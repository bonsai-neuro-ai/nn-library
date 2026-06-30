import torch
import torch.nn.functional as F
from torch import nn


class SparseAutoEncoder(nn.Module):
    """
    A single-hidden-layer sparse autoencoder (SAE), as commonly used for dictionary-learning /
    interpretability work on neural network activations.

    The architecture is a tied-bias autoencoder: the same bias vector is subtracted before
    encoding and added back after decoding, so the decoder has no bias parameter of its own. The
    hidden layer uses a ReLU nonlinearity and is trained to be sparse via an L1 penalty on the
    hidden activations (see `calculate_losses`).
    """

    def __init__(self, input_dim: int, hidden_dim: int, beta_l1: float = 0.01):
        """
        :param input_dim: dimensionality of the data being reconstructed (e.g. size of the
            activation vector being modeled).
        :param hidden_dim: number of dictionary atoms / sparse hidden units. Typically >> input_dim
            for overcomplete dictionaries.
        :param beta_l1: weight of the L1 sparsity penalty on hidden activations, relative to the
            reconstruction loss, used by `calculate_losses`.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_out_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.beta_l1 = beta_l1

    @property
    def codebook(self) -> torch.Tensor:
        """The effective dictionary atoms (decoder weight columns, shifted by the tied bias)."""
        return self.decoder.weight + self.in_out_bias[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input activations `x` (..., input_dim) to sparse hidden codes (..., hidden_dim)."""
        return F.relu(self.encoder(x - self.in_out_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input-space activations (..., input_dim) from hidden codes `z`."""
        return self.decoder(z) + self.in_out_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode `x`, returning (reconstruction, hidden_code)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def calculate_losses(self, x: torch.Tensor):
        """
        Run `x` through the autoencoder and compute the training losses.

        :param x: input activations of shape (batch, input_dim).
        :return: dict with keys "loss" (total, weighted by `beta_l1`), "reconstruction_loss"
            (mean squared error), and "sparsity_loss" (mean L1 norm of hidden codes).
        """
        x_hat, z = self(x)
        sparsity_loss = torch.mean(torch.sum(z.abs(), dim=1))
        reconstruction_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
        return {
            "loss": reconstruction_loss + self.beta_l1 * sparsity_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
        }


__all__ = [
    "SparseAutoEncoder",
]
