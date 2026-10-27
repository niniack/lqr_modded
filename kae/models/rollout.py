"""Raw student trajectories and the latent states that produced them."""

from dataclasses import dataclass

import torch


@dataclass
class StudentRollout:
    """Decoded states and the encode-once latent trajectory that produced them."""

    states: torch.Tensor
    latents: list[torch.Tensor]
