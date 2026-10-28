"""MLP encoder/decoder with a linear Koopman latent transition."""

import torch
from torch import nn

from .context_mixer import ContextMixer
from .koopman_operator import KoopmanOperator
from .rollout import KoopmanSurrogate


class KoopmanAE(KoopmanSurrogate):
    """Encode a hidden state, evolve it linearly, then decode it."""

    def __init__(
        self,
        hidden_width: int,
        latent_width: int,
        *,
        mlp_width: int = 1280,
        operator: nn.Module | None = None,
        context_width: int = 0,
        context_tokens: int = 0,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or latent_width <= 0 or mlp_width <= 0:
            raise ValueError("All widths must be positive")
        if (not isinstance(context_width, int) or isinstance(context_width, bool)
                or context_width < 0 or not isinstance(context_tokens, int)
                or isinstance(context_tokens, bool) or context_tokens < 0
                or (context_width > 0) != (context_tokens > 0)):
            raise ValueError("Context widths/tokens must be positive integers together, or both zero")

        self.hidden_width = hidden_width
        self.register_buffer("mu", torch.zeros(hidden_width, dtype=torch.float32))
        self.register_buffer("sigma", torch.ones(hidden_width, dtype=torch.float32))

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
        self.context_width = context_width
        self.context_tokens = context_tokens
        self.context_mixer = (
            ContextMixer(hidden_width, context_width, context_tokens, latent_width)
            if context_width else None
        )

    def encode(self, x: torch.Tensor, *, boundary: int | slice = 0) -> torch.Tensor:
        """Encode raw coordinates with the selected boundary's frozen affine."""
        mu, sigma = self._boundary_standardization(boundary)
        return self.encoder((x - mu) / sigma)

    def decode(self, z: torch.Tensor, *, boundary: int | slice = 0) -> torch.Tensor:
        """Decode a latent and restore raw coordinates at the selected boundary."""
        mu, sigma = self._boundary_standardization(boundary)
        return self.decoder(z) * sigma + mu
