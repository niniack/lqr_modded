"""Accumulated training diagnostics for reconstruction, rollouts, and logits."""

from collections.abc import Iterable

import torch
from torch import nn

from kae.losses import kl_divergence_loss


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


def gradient_norm_squared(parameters: Iterable[nn.Parameter]) -> torch.Tensor | None:
    """Return the sum of squared parameter-gradient norms, if any."""
    total = None
    for parameter in parameters:
        if parameter.grad is not None:
            squared = parameter.grad.detach().norm().square()
            total = squared if total is None else total + squared
    return total


def jvp_rollout(model: nn.Module, inputs: torch.Tensor, directions: torch.Tensor):
    """Return rollout and responses to cached directions in their stored order."""
    responses = []
    for direction in directions.unbind(dim=1):
        predicted, response = torch.func.jvp(model.rollout, (inputs,), (direction,))
        responses.append(response)
    return predicted, torch.stack(responses, dim=1)


class JvpMetrics:
    """Aggregate raw and relative JVP errors across examples and directions."""

    def __init__(self, output_layers: list[int]):
        self.output_layers = output_layers
        self.response_vectors = 0
        self.error_sums = torch.zeros(len(output_layers), dtype=torch.float64)
        self.teacher_sums = torch.zeros_like(self.error_sums)

    def record(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape != teacher.shape:
            raise ValueError("Student and teacher JVP response shapes must match")
        squared_error = (student - teacher).square()
        self.error_sums += squared_error.detach().sum(dim=(0, 1, 3), dtype=torch.float64).cpu()
        self.teacher_sums += teacher.detach().square().sum(
            dim=(0, 1, 3), dtype=torch.float64
        ).cpu()
        self.response_vectors += student.shape[0] * student.shape[1]
        return squared_error.mean()

    def results(self, split: str, width: int) -> dict[str, float]:
        if self.response_vectors == 0:
            return {}
        metrics = {
            f"{split}/jvp_mse": self.error_sums.sum().item()
            / (self.response_vectors * len(self.output_layers) * width),
            f"{split}/jvp_relative_error": self.error_sums.sum().item()
            / (self.teacher_sums.sum().item() + 1e-8),
        }
        for index, layer in enumerate(self.output_layers):
            metrics[f"{split}/jvp_mse/layer_{layer}"] = (
                self.error_sums[index].item() / (self.response_vectors * width)
            )
            metrics[f"{split}/jvp_relative_error/layer_{layer}"] = (
                self.error_sums[index] / (self.teacher_sums[index] + 1e-8)
            ).item()
        return metrics


class EpochDiagnostics:
    """Accumulate reconstruction, rollout, KL, and gradient diagnostics."""

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
        self.num_batches = 0
        self.gradient_groups = {"flow": model.flow, "operator": model.operator}
        if model.augmentation is not None:
            self.gradient_groups["augmentation"] = model.augmentation
        self.gradient_totals = {"total": 0.0, **dict.fromkeys(self.gradient_groups, 0.0)}
        self.loss_gradient_norms: dict[str, dict[str, float]] = {}

    def record_batch(
        self,
        inputs: torch.Tensor,
        predicted_states: torch.Tensor,
        teacher_states: torch.Tensor,
        squared_state_error: torch.Tensor,
        kl_loss: torch.Tensor | None = None,
    ) -> None:
        """Record pre-update predictions and targets, without a gradient graph."""
        with torch.no_grad():
            accumulate_rollout_stats(
                squared_state_error,
                teacher_states,
                self.state_error,
                self.state_target_sum,
                self.state_target_squares,
            )
            latent_start = self.model.encode(inputs)
            self.reconstruction_error += (
                (self.model.decode(latent_start) - inputs)
                .square()
                .sum(dtype=torch.float64)
                .cpu()
            )
            self.input_sum += inputs.sum(dim=0, dtype=torch.float64).cpu()
            self.input_squares += inputs.square().sum(dtype=torch.float64).cpu()
            predicted_latents = torch.stack(
                self.model.operator.evolve_steps(latent_start), dim=1
            )
            target_latents = self.model.encode(teacher_states)
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

    def record_loss_gradients(self, losses: dict[str, torch.Tensor | None]) -> None:
        """Measure unweighted component gradients on one pre-update batch."""
        parameters = tuple(self.model.parameters())
        groups = {
            id(parameter): name
            for name, module in self.gradient_groups.items()
            for parameter in module.parameters()
        }
        for loss_name, loss in losses.items():
            if loss is None:
                continue
            gradients = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            squares: dict[str, torch.Tensor] = {}
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    continue
                group = groups[id(parameter)]
                square = gradient.detach().square().sum()
                squares[group] = squares[group] + square if group in squares else square
            if squares:
                self.loss_gradient_norms[loss_name] = {
                    "total": torch.stack(tuple(squares.values())).sum().sqrt().item(),
                    **{name: square.sqrt().item() for name, square in squares.items()},
                }

    def record_gradients(self) -> None:
        """Record L2 gradient norms after backward, before the optimizer step."""
        group_squares = []
        for name, module in self.gradient_groups.items():
            squared = gradient_norm_squared(module.parameters())
            if squared is not None:
                self.gradient_totals[name] += squared.sqrt().item()
                group_squares.append(squared)
        if group_squares:
            self.gradient_totals["total"] += torch.stack(group_squares).sum().sqrt().item()
        self.num_batches += 1

    def results(self, split: str = "train") -> dict[str, float]:
        """Return epoch-wide MSE/FVU and optional mean pre-step gradient norms."""
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
        if self.num_batches:
            metrics.update(
                {
                    f"{split}/grad_norm/{name}": value / self.num_batches
                    for name, value in self.gradient_totals.items()
                }
            )
        for loss_name, norms in self.loss_gradient_norms.items():
            metrics.update(
                {
                    f"{split}/grad_norm_by_loss/{loss_name}/{group}": norm
                    for group, norm in norms.items()
                }
            )
        return metrics
