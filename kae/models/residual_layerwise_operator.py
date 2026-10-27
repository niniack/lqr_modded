"""Independent identity-plus-matrix Koopman transitions at each depth."""

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualTransition(nn.Module):
    def __init__(self, latent_width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(latent_width, latent_width))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + F.linear(z, self.weight)


class ResidualLayerwiseKoopmanOperator(nn.Module):
    """Apply a separate zero-initialized residual matrix at each rollout step."""

    def __init__(self, latent_width: int, steps: int = 1) -> None:
        super().__init__()
        if latent_width <= 0:
            raise ValueError("latent_width must be positive")
        if steps < 1:
            raise ValueError("steps must be positive")

        self.latent_width = latent_width
        self.steps = steps
        self.transitions = nn.ModuleList(
            _ResidualTransition(latent_width) for _ in range(steps)
        )

    def evolve_steps(
        self, z: torch.Tensor, *, start_step: int = 0, stop_step: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Return the remaining residual transitions after ``start_step``."""
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
