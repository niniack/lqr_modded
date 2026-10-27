"""Evaluate the same clean objectives on fixed training and validation documents."""

from collections import defaultdict

import torch

from kae.training.diagnostics import EpochDiagnostics
from kae.training.objectives import compute_objectives, to_device


@torch.no_grad()
def evaluate_clean(model, loader, *, device, output_layers, objective, split):
    model.eval()
    diagnostics = EpochDiagnostics(model, output_layers, objective["teacher"])
    totals = defaultdict(float)
    for source in loader:
        batch = to_device(source, device)
        # Measure every clean loss even when its optimization weight is zero.
        result = compute_objectives(model, batch, **objective, all_terms=True)
        count = batch["inputs"].shape[0]
        totals["loss"] += result.loss.item() * count
        for name, term in result.terms.items():
            totals[f"objective/{name}"] += term.item() * count
        diagnostics.record_batch(
            batch["inputs"], result.rollout.states, batch["targets"],
            (result.rollout.states - batch["targets"]).square(),
            kl_loss=result.terms["kl_divergence"], rollout=result.rollout,
            target_latents=result.target_latents,
        )
    if not diagnostics.examples:
        raise ValueError(f"{split} evaluation set is empty")
    metrics = {f"{split}/{name}": value / diagnostics.examples for name, value in totals.items()}
    metrics.update(diagnostics.results(split))
    return metrics
