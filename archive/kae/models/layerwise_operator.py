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

    def evolve_steps(
        self, z: torch.Tensor, *, start_step: int = 0, stop_step: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Return the remaining layer-specific transitions after ``start_step``."""
        states = []
        for step in range(start_step, self.steps if stop_step is None else stop_step):
            transition = self.transitions[step]
            z = transition(z)
            states.append(z)
        return tuple(states)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for transition in self.transitions:
            z = transition(z)
        return z
