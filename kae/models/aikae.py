"""Invertible-flow encoder/decoder with a Koopman latent transition."""

import torch
from torch import nn

from .affine_coupling import AffineCoupling
from .koopman_operator import KoopmanOperator


class AIKAE(nn.Module):
    """Keep physical coordinates invertible and optionally augment the latent."""

    def __init__(
        self,
        hidden_width: int,
        latent_width: int,
        *,
        mlp_width: int = 1280,
        flow_layers: int = 4,
        flow_hidden_width: int = 512,
        operator: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or hidden_width % 2:
            raise ValueError("AIKAE requires a positive, even hidden width")
        if latent_width < hidden_width:
            raise ValueError("latent_width must be at least hidden_width")
        if mlp_width <= 0 or flow_layers < 1 or flow_hidden_width <= 0:
            raise ValueError("MLP and flow widths must be positive; flow_layers must be >= 1")

        self.hidden_width = hidden_width
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        physical = x
        for layer in self.flow:
            physical = layer(physical)
        if self.augmentation is None:
            return physical
        return torch.cat((physical, self.augmentation(x)), dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        physical = z[..., : self.hidden_width]
        for layer in reversed(self.flow):
            physical = layer.inverse(physical)
        return physical

    def rollout(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoded states after each transition as ``[batch, steps, width]``."""
        states = self.operator.evolve_steps(self.encode(x))
        return torch.stack([self.decode(state) for state in states], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.operator(self.encode(x)))
