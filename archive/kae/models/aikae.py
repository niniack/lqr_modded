"""Invertible-flow encoder/decoder with a Koopman latent transition."""

import torch
from torch import nn

from .affine_coupling import AffineCoupling
from .context_mixer import ContextMixer
from .koopman_operator import KoopmanOperator
from .rollout import KoopmanSurrogate


class AIKAE(KoopmanSurrogate):
    """Keep physical coordinates invertible and optionally augment the latent."""

    def __init__(
        self,
        hidden_width: int,
        latent_width: int,
        *,
        mlp_width: int = 1280,
        flow_layers: int = 6,
        flow_hidden_width: int = 768,
        operator: nn.Module | None = None,
        context_width: int = 0,
        context_tokens: int = 0,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or hidden_width % 2:
            raise ValueError("AIKAE requires a positive, even hidden width")
        if latent_width < hidden_width:
            raise ValueError("latent_width must be at least hidden_width")
        if mlp_width <= 0 or flow_layers < 1 or flow_hidden_width <= 0:
            raise ValueError("MLP and flow widths must be positive; flow_layers must be >= 1")
        if (not isinstance(context_width, int) or isinstance(context_width, bool)
                or context_width < 0 or not isinstance(context_tokens, int)
                or isinstance(context_tokens, bool) or context_tokens < 0
                or (context_width > 0) != (context_tokens > 0)):
            raise ValueError("Context widths/tokens must be positive integers together, or both zero")

        self.hidden_width = hidden_width
        self.register_buffer("mu", torch.zeros(hidden_width, dtype=torch.float32))
        self.register_buffer("sigma", torch.ones(hidden_width, dtype=torch.float32))
        self.flow = nn.ModuleList(
            AffineCoupling(hidden_width, flow_hidden_width, flip=bool(i % 2))
            for i in range(flow_layers)
        )
        extra_width = latent_width - hidden_width
        self.augmentation = (
            nn.Sequential(
                nn.Linear(hidden_width, mlp_width),
                nn.GELU(),
                nn.Linear(mlp_width, extra_width),
            )
            if extra_width
            else None
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
        """Encode raw coordinates using the selected boundary's affine buffers."""
        mu, sigma = self._boundary_standardization(boundary)
        standardized = (x - mu) / sigma
        physical = standardized
        for layer in self.flow:
            physical = layer(physical)
        if self.augmentation is None:
            return physical
        return torch.cat((physical, self.augmentation(standardized)), dim=-1)

    def decode(self, z: torch.Tensor, *, boundary: int | slice = 0) -> torch.Tensor:
        """Invert the flow and restore raw coordinates at the selected boundary."""
        mu, sigma = self._boundary_standardization(boundary)
        physical = z[..., : self.hidden_width]
        for layer in reversed(self.flow):
            physical = layer.inverse(physical)
        return physical * sigma + mu

