"""Frozen per-boundary affine moments from an already-collected training probe."""

import torch


def fit_layerwise_standardization(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 boundary means/stds fitted with CPU FP64 population moments."""
    inputs, targets = tensors["inputs"], tensors["targets"]
    if (inputs.ndim != 2 or targets.ndim != 3 or inputs.shape[0] == 0
            or targets.shape[0] != inputs.shape[0] or targets.shape[2] != inputs.shape[1]):
        raise ValueError("Training probe must contain matching nonempty inputs [N,D] and targets [N,L,D]")
    states = torch.empty((inputs.shape[0], targets.shape[1] + 1, inputs.shape[1]),
                         device="cpu", dtype=torch.float64)
    states[:, 0].copy_(inputs.detach())
    states[:, 1:].copy_(targets.detach())
    sigma, mu = torch.std_mean(states, dim=0, correction=0)
    floor = (1e-3 * sigma.median(dim=-1, keepdim=True).values).clamp_min(1e-6)
    return mu.float(), sigma.clamp_min(floor).float()
