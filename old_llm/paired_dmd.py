"""Local, paired-response residual dynamics; independent of clean-state DMD.

For each prompt and block, perturb only the final token: dy=f(x+dx)-f(x).
Both forwards use the same full prefix, positions, mask, and copied FP32 block,
without a KV cache. The clean trajectory advances in the model's native dtype;
perturbations never propagate into another block's calibration inputs.

Random isotropic dx has norm relative_size*||x_last||. Divide each pair by the
same perturbation RMS before fitting B dx ~= dy-dx, then return A=I+B. This
weights directional responses equally rather than overweighting large states.
Differences already remove the nominal offset: fit through the origin, without
cross-prompt centering or an affine intercept. This estimates local responses,
not the effects of changed KV history during autoregressive steering.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .dmd import load_dmd_prompts


def _move(value, device, dtype=None, memo=None):
    """Move decoder arguments, retaining shared masks/rotary tensors on CPU."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        if memo is not None and key in memo:
            return memo[key][1]
        result = value.detach().to(
            device=device, dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype,
        )
        if memo is not None:
            # Retain the source too, so a later tensor cannot reuse its Python id.
            memo[key] = (value, result)
        return result
    if isinstance(value, dict):
        return {key: _move(item, device, dtype, memo) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device, dtype, memo) for item in value)
    if isinstance(value, list):
        return [_move(item, device, dtype, memo) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"Unsupported decoder argument for fixed-context pairs: {type(value).__name__}.")


def _output(block, x, kwargs):
    output = block(x, **kwargs)
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.inference_mode()
def collect_paired_responses(
    model, tokenizer, prompts, *, relative_size=0.01, batch_size=8, max_ctx=32, seed=0,
):
    """Return normalized CPU-FP32 input/output deltas [L,N,H] and scales [L,N].

    Hold one clean sequence buffer on CPU and copy each block to FP32 once.
    A native initial pass obtains the actual per-layer attention/position kwargs;
    it avoids reconstructing model-specific masks. Subsequent clean block passes
    advance the buffer, while paired evaluations leave it unperturbed. Memory
    scales as N*sequence*H plus the two L*N*H response arrays, not L*N*sequence*H.
    No model parameters, tokenizer settings, or global RNG state are changed.
    """
    if not prompts or batch_size <= 0 or max_ctx <= 0:
        raise ValueError("Provide prompts and positive batch_size/max_ctx.")
    if not math.isfinite(relative_size) or relative_size <= 0:
        raise ValueError("relative_size must be finite and positive.")
    if model.training:
        raise ValueError("Paired calibration requires model.eval().")
    if getattr(model, "is_quantized", False):
        raise ValueError("Paired FP32 block evaluation requires unquantized decoder weights.")
    layers = model.model.layers
    hidden = model.get_input_embeddings().weight.shape[1]
    count = len(prompts)
    padding_side, pad_token = tokenizer.padding_side, tokenizer.pad_token
    handles = []
    batches = []
    states = None
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad token or an EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_ctx,
        )
        if not torch.all(encoded["attention_mask"].any(dim=1)):
            raise ValueError("Every prompt must contain at least one token.")
        for index, layer in enumerate(layers):
            def capture(module, args, kwargs, index=index):
                nonlocal states
                if len(args) > 1:
                    raise ValueError("Expected decoder metadata as keyword arguments.")
                x = args[0] if args else kwargs["hidden_states"]
                if index == 0:
                    if states is None:
                        states = torch.empty(count, x.shape[1], hidden, dtype=x.dtype, device="cpu")
                    states[start:stop].copy_(x.detach().cpu())
                metadata[index] = _move(
                    {key: value for key, value in kwargs.items() if key != "hidden_states"},
                    "cpu", memo=memo,
                )
            handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        input_device = model.get_input_embeddings().weight.device
        with tqdm(total=count, desc="Preparing paired contexts", unit="prompt") as progress:
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                metadata, memo = [None] * len(layers), {}
                inputs = {key: value[start:stop].to(input_device) for key, value in encoded.items()}
                model.model(**inputs, use_cache=False, return_dict=True)
                if states is None or any(item is None for item in metadata):
                    raise RuntimeError("Paired calibration did not visit every decoder block.")
                batches.append((start, stop, metadata))
                progress.update(stop - start)
    finally:
        for handle in handles:
            handle.remove()
        tokenizer.padding_side, tokenizer.pad_token = padding_side, pad_token

    shape = (len(layers), count, hidden)
    input_deltas = torch.empty(shape, dtype=torch.float32, device="cpu")
    output_deltas = torch.empty_like(input_deltas)
    scales = torch.empty(len(layers), count, dtype=torch.float32, device="cpu")
    with tqdm(total=len(layers) * count, desc="Collecting paired responses", unit="pair") as progress:
        for index, layer in enumerate(layers):
            device = next(layer.parameters()).device
            try:
                float32_layer = copy.deepcopy(layer).to(device=device, dtype=torch.float32).eval()
            except Exception as error:
                raise RuntimeError(f"Could not copy decoder block {index} for FP32 paired responses.") from error
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            probes = torch.randn(count, hidden, generator=generator)
            probes *= math.sqrt(hidden) / probes.norm(dim=-1, keepdim=True)
            for start, stop, metadata in batches:
                native_x = states[start:stop].to(device=device)
                native_kwargs = _move(metadata[index], device)
                x = native_x.float()
                kwargs = _move(metadata[index], device, torch.float32)
                scale = relative_size * x[:, -1].norm(dim=-1) / math.sqrt(hidden)
                if not torch.all(torch.isfinite(scale) & (scale > 0)):
                    raise ValueError(f"Nonfinite or zero final-token norm at block {index}.")
                with torch.autocast(device_type=device.type, enabled=False):
                    clean_output = _output(float32_layer, x, kwargs)[:, -1].clone()
                    perturbed = x.clone()
                    perturbed[:, -1] += probes[start:stop].to(device) * scale[:, None]
                    # Record actual FP32 additions, rather than requested dx before rounding.
                    dx = (perturbed[:, -1] - x[:, -1]) / scale[:, None]
                    dy = (_output(float32_layer, perturbed, kwargs)[:, -1] - clean_output) / scale[:, None]
                if not torch.isfinite(dx).all() or not torch.isfinite(dy).all():
                    raise ValueError(f"Nonfinite paired responses at block {index}.")
                input_deltas[index, start:stop].copy_(dx.cpu())
                output_deltas[index, start:stop].copy_(dy.cpu())
                scales[index, start:stop].copy_(scale.cpu())
                # Only the unperturbed native block output seeds the next layer.
                if index + 1 < len(layers):
                    states[start:stop].copy_(_output(layer, native_x, native_kwargs).cpu())
                metadata[index] = None
                progress.set_postfix(layer=f"{index + 1}/{len(layers)}", refresh=False)
                progress.update(stop - start)
            del float32_layer, probes, native_x, native_kwargs, x, kwargs, clean_output, perturbed, dx, dy
    return {"input_deltas": input_deltas, "output_deltas": output_deltas, "scales": scales}


def fit_paired_dmd(responses, *, alpha=1e-3):
    """Fit full-dimensional, origin-preserving residual ridge; return raw FP32 A.

    Minimize mean ||(dy-dx)-B dx||^2 + alpha*||B||_F^2 per layer. Inputs/outputs
    share the recorded scalar normalization; no coordinate standardization,
    centering, reduced rank, or affine term is introduced.
    """
    x, y = responses["input_deltas"], responses["output_deltas"]
    if x.ndim != 3 or x.shape != y.shape or min(x.shape) < 1:
        raise ValueError("Paired deltas must have matching (layers, samples, hidden) shapes.")
    layers, count, hidden = x.shape
    if count < hidden:
        raise ValueError(f"Full-dimensional paired fitting requires N >= H; got N={count}, H={hidden}.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Paired alpha must be finite and positive.")
    dynamics = torch.empty(layers, hidden, hidden, dtype=torch.float32)
    for layer in tqdm(range(layers), desc="Fitting paired dynamics", unit="layer"):
        inputs = x[layer].detach().to(device="cpu", dtype=torch.float64)
        residual = y[layer].detach().to(device="cpu", dtype=torch.float64) - inputs
        if not torch.isfinite(inputs).all() or not torch.isfinite(residual).all():
            raise ValueError(f"Nonfinite paired responses at block {layer}.")
        covariance = inputs.T @ inputs / count
        covariance.diagonal().add_(alpha)
        cross_covariance = inputs.T @ residual / count
        correction = torch.cholesky_solve(cross_covariance, torch.linalg.cholesky(covariance)).T
        correction.diagonal().add_(1)
        dynamics[layer].copy_(correction)
    if not torch.isfinite(dynamics).all():
        raise ValueError("Nonfinite paired dynamics.")
    return {"A": dynamics}


def load_or_fit_paired_dmd(
    model, tokenizer, cache_dir, *, sample_size=None, relative_size=0.01, alpha=1e-3,
    batch_size=8, max_ctx=32, seed=0,
):
    """Reuse separate paired-response/fit caches; default to 2H mixed prompts.

    Fitting is full-dimensional (N >= H). Changing alpha reuses pairs without
    model forwards; changing perturbation size requires new responses. Existing
    clean-state DMD caches are never read or overwritten by this helper.
    """
    hidden = model.get_input_embeddings().weight.shape[1]
    sample_size = 2 * hidden if sample_size is None else sample_size
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < hidden:
        raise ValueError(f"Full-dimensional paired fitting requires integer sample_size >= H={hidden}.")
    if not math.isfinite(relative_size) or relative_size <= 0:
        raise ValueError("Paired relative_size must be finite and positive.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Paired alpha must be finite and positive.")
    if batch_size <= 0 or max_ctx <= 0:
        raise ValueError("Paired batch_size and max_ctx must be positive.")
    dtype_tag = str(next(model.parameters()).dtype).removeprefix("torch.")
    tag = f"mixed_n{sample_size}_eps{relative_size}_ctx{max_ctx}_seed{seed}_{dtype_tag}_fp32"
    cache_dir = Path(cache_dir)
    response_path = cache_dir / f"responses_paired_{tag}.pt"
    fit_path = cache_dir / f"A_paired_{tag}_alpha{alpha}.pt"
    if fit_path.exists():
        tqdm.write(f"Loading paired fit: {fit_path.name}")
        return torch.load(fit_path, map_location="cpu", weights_only=True), fit_path
    if response_path.exists():
        tqdm.write(f"Loading paired responses: {response_path.name}")
        responses = torch.load(response_path, map_location="cpu", weights_only=True)
    else:
        prompts = load_dmd_prompts(sample_size, seed=seed)
        responses = collect_paired_responses(
            model, tokenizer, prompts, relative_size=relative_size,
            batch_size=batch_size, max_ctx=max_ctx, seed=seed,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(responses, response_path)
    expected_shape = (len(model.model.layers), sample_size, hidden)
    if any(responses[key].shape != expected_shape for key in ("input_deltas", "output_deltas")):
        raise ValueError(f"Paired deltas must have shape {expected_shape}.")
    fit = fit_paired_dmd(responses, alpha=alpha)
    torch.save(fit, fit_path)
    tqdm.write(f"Saved paired fit: {fit_path.name}")
    return fit, fit_path
