"""Independent linear Koopman transitions for successive teacher layers."""

import torch
from torch import nn


class LayerwiseKoopmanOperator(nn.Module):
    """Use one learned latent transition for each rollout step."""

    def __init__(self, latent_width: int, steps: int = 1) -> None:
        super().__init__()
        if latent_width <= 0:
            raise ValueError("latent_width must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        self.latent_width = latent_width
        self.steps = steps
        self.transitions = nn.ModuleList(
            nn.Linear(latent_width, latent_width, bias=False) for _ in range(steps)
        )
        for transition in self.transitions:
            nn.init.eye_(transition.weight)

    def evolve_steps(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the latent state after each layer-specific transition."""
        states = []
        for transition in self.transitions:
            z = transition(z)
            states.append(z)
        return tuple(states)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for transition in self.transitions:
            z = transition(z)
        return z
