"""Affine coupling layer used by the AIKAE physical-state flow."""

import torch
from torch import nn


def _normalize_condition(x: torch.Tensor) -> torch.Tensor:
    """Normalize conditioner features while preserving differentiable statistics."""
    values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(variance + 1e-5)).to(x.dtype)


class AffineCoupling(nn.Module):
    """Transform one half of a vector while conditioning on the other half."""

    def __init__(self, width: int, hidden_width: int, *, flip: bool = False) -> None:
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError("Affine coupling requires a positive, even width")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.flip = flip
        half = width // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 2 * half),
        )
        # Initialize the flow and its inverse as exact identities.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _shift_scale(self, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, raw_scale = self.net(_normalize_condition(fixed)).chunk(2, dim=-1)
        return shift, 0.5 * torch.tanh(raw_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        fixed, moving = (right, left) if self.flip else (left, right)
        shift, log_scale = self._shift_scale(fixed)
        moving = moving * torch.exp(log_scale) + shift
        return torch.cat((moving, fixed) if self.flip else (fixed, moving), dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        left, right = z.chunk(2, dim=-1)
        fixed, moving = (right, left) if self.flip else (left, right)
        shift, log_scale = self._shift_scale(fixed)
        moving = (moving - shift) * torch.exp(-log_scale)
        return torch.cat((moving, fixed) if self.flip else (fixed, moving), dim=-1)
