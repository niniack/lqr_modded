"""Koopman dynamics with a shared modal basis and per-step 2x2 blocks."""

import torch
from torch import nn


class SharedModesKoopmanOperator(nn.Module):
    """Apply per-step growth/rotation blocks in one learned modal basis."""

    def __init__(self, latent_width: int, steps: int = 1) -> None:
        super().__init__()
        if latent_width <= 0 or latent_width % 2:
            raise ValueError("shared modes require a positive, even latent width")
        if steps < 1:
            raise ValueError("steps must be positive")

        self.latent_width = latent_width
        self.steps = steps
        lower = torch.tril_indices(latent_width, latent_width, offset=-1)
        upper = torch.triu_indices(latent_width, latent_width, offset=1)
        self.register_buffer("_lower_rows", lower[0], persistent=False)
        self.register_buffer("_lower_cols", lower[1], persistent=False)
        self.register_buffer("_upper_rows", upper[0], persistent=False)
        self.register_buffer("_upper_cols", upper[1], persistent=False)
        self.register_buffer("_diagonal", torch.arange(latent_width), persistent=False)
        self.register_buffer("_identity", torch.eye(latent_width), persistent=False)
        self.mode_lower = nn.Parameter(torch.zeros(lower.shape[1]))
        self.mode_upper = nn.Parameter(torch.zeros(upper.shape[1]))
        self.mode_log_diagonal = nn.Parameter(torch.zeros(latent_width))

        blocks = torch.zeros(steps, latent_width // 2, 2)
        blocks[:, :, 0] = 1
        self.mode_blocks = nn.Parameter(blocks)

    def operator_matrix(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the invertible lower/upper basis and return its modal blocks."""
        lower = self._identity.index_put(
            (self._lower_rows, self._lower_cols), self.mode_lower
        )
        upper = self._identity.index_put(
            (self._upper_rows, self._upper_cols), self.mode_upper
        )
        upper = upper.index_put(
            (self._diagonal, self._diagonal), self.mode_log_diagonal.exp()
        )
        return lower, upper, self.mode_blocks

    @staticmethod
    def _to_modes(z: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        z = torch.linalg.solve_triangular(
            lower, z.mT, upper=False, unitriangular=True
        )
        return torch.linalg.solve_triangular(upper, z, upper=True)

    @staticmethod
    def _from_modes(z: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return (lower @ (upper @ z)).mT

    @staticmethod
    def _apply_block(latent: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        first, second = latent.unflatten(0, (-1, 2)).unbind(dim=1)
        real, imag = block.unbind(dim=-1)
        return torch.stack(
            (real[:, None] * first - imag[:, None] * second,
             imag[:, None] * first + real[:, None] * second),
            dim=1,
        ).flatten(0, 1)

    def evolve_steps(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the latent state after each modal transition."""
        lower, upper, blocks = self.operator_matrix()
        latent = self._to_modes(z, lower, upper)
        states = []
        for block in blocks:
            latent = self._apply_block(latent, block)
            states.append(self._from_modes(latent, lower, upper))
        return tuple(states)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lower, upper, blocks = self.operator_matrix()
        latent = self._to_modes(z, lower, upper)
        for block in blocks:
            latent = self._apply_block(latent, block)
        return self._from_modes(latent, lower, upper)
