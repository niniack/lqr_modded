"""Invertible-flow encoder/decoder with a Koopman latent transition."""

import torch
from torch import nn

from .affine_coupling import AffineCoupling
from .context_mixer import ContextMixer
from .koopman_operator import KoopmanOperator
from .rollout import StudentRollout


class AIKAE(nn.Module):
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

    @torch.no_grad()
    def set_standardization(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Own frozen FP32 affine buffers shared across or indexed by boundaries."""
        shapes = ((self.hidden_width,), (self.operator.steps + 1, self.hidden_width))
        if mu.shape not in shapes or sigma.shape != mu.shape:
            raise ValueError(
                "Standardization requires matching [hidden_width] or "
                "[operator_steps + 1, hidden_width] buffers"
            )
        mu = mu.detach().to(device=self.mu.device, dtype=torch.float32, copy=True)
        sigma = sigma.detach().to(device=self.sigma.device, dtype=torch.float32, copy=True)
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
            raise ValueError("Standardization buffers must be finite")
        if not bool((sigma > 0).all()):
            raise ValueError("Standardization scales must be positive")
        self.mu = mu
        self.sigma = sigma

    def _boundary_standardization(
        self, boundary: int | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(boundary, (int, slice)):
            raise TypeError("Standardization boundary must be an int or slice")
        if isinstance(boundary, int) and not 0 <= boundary <= self.operator.steps:
            raise ValueError("Standardization boundary must be in [0, operator_steps]")
        if self.mu.ndim == 1:
            return self.mu, self.sigma
        return self.mu[boundary], self.sigma[boundary]

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

    def _context_forcing(
        self, x0: torch.Tensor, context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.context_mixer is None:
            if context is not None or context_mask is not None:
                raise ValueError("Context tensors require a context-enabled student")
            return None
        if context is None or context_mask is None:
            raise ValueError("Context-enabled students require context and context_mask")
        if (context.shape != (x0.shape[0], self.context_tokens, self.hidden_width)
                or context.dtype != torch.float32 or context.device != x0.device):
            raise ValueError("Context must be FP32 [batch, context_tokens, hidden_width] on the input device")
        if (context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool
                or context_mask.device != x0.device):
            raise ValueError("Context mask must be bool [batch, context_tokens] on the input device")
        mu, sigma = self._boundary_standardization(0)
        return self.context_mixer((context - mu) / sigma, context_mask)

    def _evolve(
        self, z: torch.Tensor, forcing: torch.Tensor | None, *, start_step: int = 0
    ) -> tuple[torch.Tensor, ...]:
        """Add fixed context only at the first absolute transition."""
        if forcing is None or start_step > 0:
            return self.operator.evolve_steps(z, start_step=start_step)
        first = self.operator.evolve_steps(z, stop_step=1)[0] + forcing
        if self.operator.steps == 1:
            return (first,)
        return (first, *self.operator.evolve_steps(first, start_step=1))

    def _rollout_details(
        self, x0: torch.Tensor, forcing: torch.Tensor | None
    ) -> StudentRollout:
        z0 = self.encode(x0)
        latents = [z0, *self._evolve(z0, forcing)]
        states = torch.stack(
            [self.decode(z, boundary=j) for j, z in enumerate(latents[1:], start=1)],
            dim=1,
        )
        return StudentRollout(states=states, latents=latents)

    def rollout_details(
        self, x0: torch.Tensor, *, context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> StudentRollout:
        """Encode once, condition the first transition, and retain clean latents."""
        return self._rollout_details(x0, self._context_forcing(x0, context, context_mask))

    def rollout(
        self, x: torch.Tensor, *, context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw decoded states shaped ``[batch, steps, hidden_width]``."""
        return self.rollout_details(x, context=context, context_mask=context_mask).states

    def forward(
        self, x: torch.Tensor, *, context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the final raw state after all configured Koopman steps."""
        forcing = self._context_forcing(x, context, context_mask)
        encoded = self.encode(x)
        latent = self.operator(encoded) if forcing is None else self._evolve(encoded, forcing)[-1]
        return self.decode(latent, boundary=self.operator.steps)
