"""Muon matrix weights, Adam biases, gradient clipping, and cosine warmup."""

import math

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR


MUON_WEIGHT_MULTIPLIER = 1.5
MUON_MOMENTUM = 0.95


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


def build_optimizers(
    model, peak_learning_rate: float, *, muon_lr_multiplier: float = MUON_WEIGHT_MULTIPLIER,
) -> list[Optimizer]:
    """Scale Muon matrix LR by the multiplier; keep Adam non-matrix LR at the base peak."""
    biases, other, weights = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias"):
            biases.append(parameter)
        elif parameter.ndim == 2:
            weights.append(parameter)
        else:
            other.append(parameter)

    if not weights or not (biases or other):
        raise ValueError("Expected trainable matrix weights and at least one bias or non-matrix parameter")

    adam_groups = []
    if biases:
        adam_groups.append({"params": biases, "lr": peak_learning_rate, "name": "adam_biases"})
    if other:
        adam_groups.append({"params": other, "lr": peak_learning_rate, "name": "adam_nonmatrix"})
    adam = Adam(adam_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=True)
    muon = torch.optim.Muon(
        [{"params": weights, "lr": peak_learning_rate * muon_lr_multiplier,
          "name": "muon_weights"}],
        lr=peak_learning_rate * muon_lr_multiplier,
        weight_decay=0.0, momentum=MUON_MOMENTUM, nesterov=True,
        ns_steps=5, adjust_lr_fn="match_rms_adamw",
    )
    return [adam, muon]


class Optimization:
    def __init__(self, model, recipe, total_steps):
        self.model = model
        self.grad_clip = recipe.grad_clip
        self.warmup_steps = math.ceil(recipe.warmup_fraction * total_steps)
        self.optimizers = build_optimizers(
            model, recipe.learning_rate, muon_lr_multiplier=recipe.muon_lr_multiplier)
        self.schedulers = [
            cosine_warmup_scheduler(optimizer, total_steps, self.warmup_steps)
            for optimizer in self.optimizers
        ]
        self.update_count = 0

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @property
    def learning_rate(self):
        """The first group's rate; initially this is the unscaled Adam bias rate."""
        return self.param_groups[0]["lr"]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        """Apply existing gradients and report the rates used for this update."""
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        for optimizer in self.optimizers:
            optimizer.step()
        groups = self.param_groups
        used_rates = [(group.get("name", f"group_{index}"), float(group["lr"]))
                      for index, group in enumerate(groups)]

        for scheduler in self.schedulers:
            scheduler.step()
        self.update_count += 1

        metrics = {"grad_norm_pre_clip": norm.item(), "learning_rate": used_rates[0][1]}
        for name, rate in used_rates:
            metrics[f"learning_rate/{name}"] = rate
        return metrics

    def state_dict(self):
        return {"optimizer": [optimizer.state_dict() for optimizer in self.optimizers],
                "scheduler": [scheduler.state_dict() for scheduler in self.schedulers],
                "update_count": self.update_count}

    def load_state_dict(self, state):
        for optimizer, saved in zip(self.optimizers, state["optimizer"], strict=True):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.schedulers, state["scheduler"], strict=True):
            scheduler.load_state_dict(saved)
        self.update_count = state["update_count"]
