"""Optimizer schedules used by AIKAE training."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_steps: int
) -> LambdaLR:
    """Warm linearly to the optimizer's peak LR, then decay to zero."""
    if total_steps < 1 or not 1 <= warmup_steps <= total_steps:
        raise ValueError("Require at least one step and 1 <= warmup_steps <= total_steps")

    def multiplier(step_index: int) -> float:
        update = step_index + 1
        if update <= warmup_steps:
            return update / warmup_steps
        if update >= total_steps:
            return 0.0
        progress = (update - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)
