"""MLP encoder/decoder with a linear Koopman latent transition."""

import torch
from torch import nn

from .koopman_operator import KoopmanOperator


class KoopmanAE(nn.Module):
    """Encode a hidden state, evolve it linearly, then decode it."""

    def __init__(
        self,
        hidden_width: int,
        latent_width: int,
        *,
        mlp_width: int = 1280,
        operator: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or latent_width <= 0 or mlp_width <= 0:
            raise ValueError("All widths must be positive")

        self.encoder = nn.Sequential(
            nn.Linear(hidden_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, latent_width),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_width, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, hidden_width),
        )
        self.operator = (
            KoopmanOperator(latent_width) if operator is None else operator
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.operator(self.encode(x)))
