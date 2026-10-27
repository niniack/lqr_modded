"""A learned linear transition applied repeatedly in latent space."""

import torch
from torch import nn


class KoopmanOperator(nn.Module):
    """Use one shared transition matrix for the configured number of steps."""

    def __init__(self, latent_width: int, steps: int = 1) -> None:
        super().__init__()
        if latent_width <= 0:
            raise ValueError("latent_width must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        self.latent_width = latent_width
        self.steps = steps
        self.transition = nn.Linear(latent_width, latent_width, bias=False)
        nn.init.eye_(self.transition.weight)

    def evolve_steps(
        self, z: torch.Tensor, *, start_step: int = 0, stop_step: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Return latent transitions from the requested absolute boundary."""
        states = []
        for _ in range(start_step, self.steps if stop_step is None else stop_step):
            z = self.transition(z)
            states.append(z)
        return tuple(states)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            z = self.transition(z)
        return z
