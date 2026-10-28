"""Raw reconstruction, state, latent, later-start rollout, and next-token KL objectives."""

from dataclasses import dataclass

import torch

from kae.losses import kl_divergence_loss, mse_loss


def objective_weights(config):
    return {
        "reconstruction_mse": config.reconstruction_weight,
        "state_mse": config.state_weight,
        "latent_mse": config.latent_weight,
        "kl_divergence": config.kl_weight,
        "multistart_state_mse": config.multistart_state_weight,
        "multistart_latent_mse": config.multistart_latent_weight,
    }


@dataclass
class Objectives:
    loss: torch.Tensor
    terms: dict[str, torch.Tensor]
    rollout: object
    target_latents: torch.Tensor | None
    reconstructed_inputs: torch.Tensor | None


def compute_objectives(model, batch, *, teacher, weights, all_terms=False):
    """Encode once and propagate; evaluation also measures zero-weight losses."""
    context = {key: batch[key] for key in ("context", "context_mask") if key in batch}
    rollout = model.rollout_details(batch["inputs"], **context)
    terms = {}
    reconstructed_inputs = None
    if all_terms or weights["reconstruction_mse"]:
        # Reuse z0: reconstruction trains the encoder/decoder, not the operator.
        reconstructed_inputs = model.decode(rollout.latents[0], boundary=0)
        terms["reconstruction_mse"] = mse_loss(reconstructed_inputs, batch["inputs"])
    if all_terms or weights["state_mse"]:
        terms["state_mse"] = mse_loss(rollout.states, batch["targets"])
    if all_terms or weights["kl_divergence"]:
        if teacher is None:
            raise ValueError("Clean KL requires a frozen final-boundary teacher head")
        with torch.no_grad():
            target_logits = teacher(batch["targets"][:, -1])
        terms["kl_divergence"] = kl_divergence_loss(teacher(rollout.states[:, -1]), target_logits)
    target_latents = None
    if all_terms or weights["latent_mse"] or weights["multistart_latent_mse"]:
        target_latents = model.encode(batch["targets"], boundary=slice(1, None))
    if all_terms or weights["latent_mse"]:
        terms["latent_mse"] = mse_loss(
            torch.stack(rollout.latents[1:], dim=1), target_latents)
    multistart_active = bool(weights["multistart_state_mse"] or weights["multistart_latent_mse"])
    if multistart_active or (all_terms and model.operator.steps > 1):
        terms.update(multistart_losses(
            model, batch["targets"],
            state=all_terms or bool(weights["multistart_state_mse"]),
            latent=all_terms or bool(weights["multistart_latent_mse"]),
            target_latents=target_latents,
        ))
    # Zero-weight diagnostic terms must not contribute to the training gradient.
    loss = sum(weights[name] * term for name, term in terms.items() if weights[name])
    return Objectives(loss, terms, rollout, target_latents, reconstructed_inputs)


def to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def multistart_losses(model, teacher_states, *, state, latent, target_latents=None):
    """Average raw state/latent MSE equally over all later-start/end pairs.

    ``teacher_states`` is [B, steps, D], excluding the original input boundary.
    Encode each teacher start once; never reset a propagated latent to a target.
    Per-boundary affine transforms and absolute operator indices stay aligned.
    """
    steps = teacher_states.shape[1]
    if steps < 2:
        raise ValueError("Clean multi-start losses require at least two output boundaries")
    if target_latents is None:
        # State-only supervision needs the starts, not the final endpoint encoding.
        states = teacher_states if latent else teacher_states[:, :-1]
        boundaries = slice(1, None) if latent else slice(1, -1)
        target_latents = model.encode(states, boundary=boundaries)

    terms = {}
    for start in range(1, steps):
        predicted_latents = torch.stack(model.operator.evolve_steps(
            target_latents[:, start - 1], start_step=start), dim=1)
        horizons = steps - start
        if state:
            predicted_states = model.decode(
                predicted_latents, boundary=slice(start + 1, None))
            error = mse_loss(predicted_states, teacher_states[:, start:])
            name = "multistart_state_mse"
            terms[name] = terms.get(name, 0) + horizons * error
        if latent:
            error = mse_loss(predicted_latents, target_latents[:, start:])
            name = "multistart_latent_mse"
            terms[name] = terms.get(name, 0) + horizons * error

    pairs = steps * (steps - 1) // 2
    return {name: total / pairs for name, total in terms.items()}
