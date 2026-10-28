"""Raw state/latent accuracy and fixed-training-probe gradient diagnostics."""


from itertools import combinations
import math

import torch
from torch import nn

from kae.losses import kl_divergence_loss
from kae.models import KoopmanAE


def accumulate_rollout_stats(
    squared_error: torch.Tensor,
    target: torch.Tensor,
    error_per_layer: torch.Tensor,
    target_sum: torch.Tensor,
    target_square_per_layer: torch.Tensor,
) -> None:
    """Accumulate squared errors and teacher moments across examples."""
    error_per_layer += squared_error.detach().sum(dim=(0, 2), dtype=torch.float64).cpu()
    target_sum += target.detach().sum(dim=0, dtype=torch.float64).cpu()
    target_square_per_layer += target.detach().square().sum(
        dim=(0, 2), dtype=torch.float64
    ).cpu()


def fvu(
    squared_error: torch.Tensor,
    target_sum: torch.Tensor,
    target_square_sum: torch.Tensor,
    examples: int,
) -> float:
    """SSE divided by teacher variation around each feature's epoch-wide mean."""
    centered_variation = (
        target_square_sum.sum().item() - target_sum.square().sum().item() / examples
    )
    return squared_error.sum().item() / (max(centered_variation, 0.0) + 1e-8)




class EpochDiagnostics:
    """Accumulate raw reconstruction, rollout, latent, and clean KL diagnostics."""

    def __init__(self, model: nn.Module, output_layers: list[int], teacher: nn.Module | None):
        self.model = model
        self.teacher = teacher
        self.output_layers = output_layers
        steps = len(output_layers)
        self.reconstruction_error = torch.zeros((), dtype=torch.float64)
        self.input_sum = torch.zeros(model.hidden_width, dtype=torch.float64)
        self.input_squares = torch.zeros((), dtype=torch.float64)
        self.state_error = torch.zeros(steps, dtype=torch.float64)
        self.state_target_sum = torch.zeros(steps, model.hidden_width, dtype=torch.float64)
        self.state_target_squares = torch.zeros(steps, dtype=torch.float64)
        self.latent_error = torch.zeros(steps, dtype=torch.float64)
        self.latent_target_sum = torch.zeros(steps, model.operator.latent_width, dtype=torch.float64)
        self.latent_target_squares = torch.zeros(steps, dtype=torch.float64)
        self.examples = 0
        self.kl_total = 0.0

    def record_batch(
        self,
        inputs: torch.Tensor,
        predicted_states: torch.Tensor,
        teacher_states: torch.Tensor,
        squared_state_error: torch.Tensor,
        kl_loss: torch.Tensor | None = None,
        rollout=None,
        target_latents: torch.Tensor | None = None,
        reconstructed_inputs: torch.Tensor | None = None,
    ) -> None:
        """Record predictions and targets without retaining a gradient graph."""
        with torch.no_grad():
            accumulate_rollout_stats(
                squared_state_error,
                teacher_states,
                self.state_error,
                self.state_target_sum,
                self.state_target_squares,
            )
            latent_start = self.model.encode(inputs, boundary=0) if rollout is None else rollout.latents[0]
            if reconstructed_inputs is None:
                reconstructed_inputs = self.model.decode(latent_start, boundary=0)
            self.reconstruction_error += (
                (reconstructed_inputs - inputs)
                .square()
                .sum(dtype=torch.float64)
                .cpu()
            )
            self.input_sum += inputs.sum(dim=0, dtype=torch.float64).cpu()
            self.input_squares += inputs.square().sum(dtype=torch.float64).cpu()
            predicted_latents = (
                torch.stack(self.model.operator.evolve_steps(latent_start), dim=1)
                if rollout is None else torch.stack(rollout.latents[1:], dim=1)
            )
            if target_latents is None:
                target_latents = self.model.encode(teacher_states, boundary=slice(1, None))
            accumulate_rollout_stats(
                (predicted_latents - target_latents).square(),
                target_latents,
                self.latent_error,
                self.latent_target_sum,
                self.latent_target_squares,
            )
            if kl_loss is not None:
                self.kl_total += kl_loss.detach().item() * inputs.shape[0]
            elif self.teacher is not None:
                predicted_logits = self.teacher(predicted_states[:, -1])
                target_logits = self.teacher(teacher_states[:, -1])
                self.kl_total += (
                    kl_divergence_loss(predicted_logits, target_logits).item()
                    * inputs.shape[0]
                )
        self.examples += inputs.shape[0]


    def results(self, split: str = "train") -> dict[str, float]:
        """Return epoch-wide raw-coordinate MSE/FVU and clean KL."""
        steps = len(self.output_layers)
        state_width = self.state_target_sum.shape[-1]
        latent_width = self.latent_target_sum.shape[-1]
        metrics = {
            f"{split}/reconstruction_mse": self.reconstruction_error.item()
            / (self.examples * state_width),
            f"{split}/reconstruction_fvu": fvu(
                self.reconstruction_error,
                self.input_sum,
                self.input_squares,
                self.examples,
            ),
            f"{split}/state_mse_first": self.state_error[0].item() / (self.examples * state_width),
            f"{split}/state_fvu_first": fvu(
                self.state_error[0],
                self.state_target_sum[0],
                self.state_target_squares[0],
                self.examples,
            ),
            f"{split}/state_rollout_mse": self.state_error.sum().item()
            / (self.examples * steps * state_width),
            f"{split}/state_rollout_fvu": fvu(
                self.state_error,
                self.state_target_sum,
                self.state_target_squares,
                self.examples,
            ),
            f"{split}/latent_mse": self.latent_error.sum().item()
            / (self.examples * steps * latent_width),
            f"{split}/latent_fvu": fvu(
                self.latent_error,
                self.latent_target_sum,
                self.latent_target_squares,
                self.examples,
            ),
        }
        for index, layer in enumerate(self.output_layers):
            metrics[f"{split}/state_mse/layer_{layer}"] = (
                self.state_error[index].item() / (self.examples * state_width)
            )
            metrics[f"{split}/state_fvu/layer_{layer}"] = fvu(
                self.state_error[index],
                self.state_target_sum[index],
                self.state_target_squares[index],
                self.examples,
            )
        if self.teacher is not None:
            metrics[f"{split}/kl_divergence"] = self.kl_total / self.examples
        return metrics


def fixed_probe_gradients(model, batch, *, objective):
    """Measure counterfactual objective gradients on the same pre-update probe."""
    from kae.training.objectives import compute_objectives

    model.train()
    result = compute_objectives(model, batch, **objective, all_terms=True)
    if isinstance(model, KoopmanAE):
        groups = {"encoder": model.encoder, "decoder": model.decoder, "operator": model.operator}
    else:
        groups = {"flow": model.flow, "operator": model.operator}
        if model.augmentation is not None:
            groups["augmentation"] = model.augmentation
    if model.context_mixer is not None:
        groups["context"] = model.context_mixer
    metrics = loss_gradient_stats(result.terms, weights=objective["weights"],
                                  model=model, groups=groups)
    return {f"train/{name}": value for name, value in metrics.items()}


def loss_gradient_stats(
    losses: dict[str, torch.Tensor | None],
    *,
    weights: dict[str, float],
    model: nn.Module,
    groups: dict[str, nn.Module],
) -> dict[str, float]:
    """Return flat raw/weighted norms and unweighted pairwise cosines.

    Missing parameter gradients count as zero, including disjoint supports.
    Zero-norm objectives have defined zero norms but no defined cosine.
    The caller's loss graph is retained for the ordinary training backward.
    Retain diagnostic gradients on CPU rather than keeping a complete
    parameter-gradient set per objective on the training GPU.
    """
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    group_members = {
        name: {id(parameter) for parameter in module.parameters()}
        for name, module in groups.items()
    }
    parameter_groups = tuple(
        ("total", *(name for name, members in group_members.items() if id(parameter) in members))
        for parameter in parameters
    )
    group_names = ("total", *groups)
    gradients_by_loss = {}
    norms_by_loss = {}
    metrics = {}
    for loss_name, loss in losses.items():
        if loss is None:
            continue
        if parameters and loss.requires_grad:
            gradients = tuple(
                gradient.detach().cpu() if gradient is not None else None
                for gradient in torch.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                )
            )
        else:
            gradients = (None,) * len(parameters)
        gradients_by_loss[loss_name] = gradients
        squares = dict.fromkeys(group_names, None)
        for names, gradient in zip(parameter_groups, gradients):
            if gradient is None:
                continue
            square = gradient.norm().square()
            for name in names:
                squares[name] = square if squares[name] is None else squares[name] + square
        norms = {
            name: math.sqrt(max(square.item(), 0.0)) if square is not None else 0.0
            for name, square in squares.items()
        }
        norms_by_loss[loss_name] = norms
        for name, norm in norms.items():
            metrics[f"grad_norm_by_loss/{loss_name}/{name}"] = norm
            metrics[f"grad_norm_weighted_by_loss/{loss_name}/{name}"] = abs(weights[loss_name]) * norm

    for loss_a, loss_b in combinations(gradients_by_loss, 2):
        dots = dict.fromkeys(group_names, None)
        for names, gradient_a, gradient_b in zip(
            parameter_groups, gradients_by_loss[loss_a], gradients_by_loss[loss_b]
        ):
            if gradient_a is None or gradient_b is None:
                continue
            dot = (
                gradient_a * gradient_b
                if gradient_a.ndim == 0
                else torch.linalg.vecdot(gradient_a, gradient_b).sum()
            )
            for name in names:
                dots[name] = dot if dots[name] is None else dots[name] + dot
        for name, dot in dots.items():
            norm_a = norms_by_loss[loss_a][name]
            norm_b = norms_by_loss[loss_b][name]
            if norm_a == 0.0 or norm_b == 0.0:
                continue
            # Disjoint supports have a zero dot product, not an undefined angle.
            cosine = (dot.item() if dot is not None else 0.0) / (norm_a * norm_b)
            metrics[f"grad_cosine/{loss_a}_vs_{loss_b}/{name}"] = min(max(cosine, -1.0), 1.0)
    return metrics
