"""Masked fixed-slot MLP-Mixer for raw suffix-boundary prefix context."""

import torch
from torch import nn


class _MixerBlock(nn.Module):
    def __init__(self, tokens: int, width: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, min(tokens, 128)), nn.GELU(),
            nn.Linear(min(tokens, 128), tokens),
        )
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.token_norm(x).masked_fill(~mask, 0)
        x = (x + self.token_mlp(normalized.transpose(1, 2)).transpose(1, 2)).masked_fill(~mask, 0)
        return (x + self.channel_mlp(self.channel_norm(x))).masked_fill(~mask, 0)


class ContextMixer(nn.Module):
    """Mix preceding-token slots, pool valid tokens, and emit first-step forcing."""

    def __init__(self, hidden_width: int, width: int, tokens: int, latent_width: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_width, width)
        self.blocks = nn.ModuleList(_MixerBlock(tokens, width) for _ in range(2))
        self.output_projection = nn.Linear(width, latent_width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        mask = context_mask.unsqueeze(-1)
        x = self.input_projection(context.masked_fill(~mask, 0)).masked_fill(~mask, 0)
        for block in self.blocks:
            x = block(x, mask)
        counts = context_mask.sum(dim=1, keepdim=True)
        pooled = x.sum(dim=1) / counts.clamp_min(1)
        return self.output_projection(pooled).masked_fill(counts == 0, 0)
