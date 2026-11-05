"""Per-block, centered residual DMD on final prompt tokens for notebooks and benchmarks.

The linear-kernel PCR estimator follows kooplearn's fluid-flow DMD example:
https://kooplearn.readthedocs.io/latest/examples/fluid_flow_dmd.html

Fit r = x_next - x after centering both x and r, then return A = I + B
in the controller's raw, column-vector coordinates. Optional standardization
uses the block INPUT standard deviation for both x and r, so the identity prior
stays identity when mapping back. This changes fitting, not LQR costs/coordinates.
The fitted affine intercept is unnecessary for the deviation model used by LQR.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from kooplearn.kernel import KernelRidge
from kooplearn.linear_model import Ridge
from tqdm.auto import tqdm

from .directions import SENTENCES_CSV, ContrastiveBuilder


def load_dmd_prompts(sample_size, *, seed=0, csv_path=SENTENCES_CSV):
    """Sample unique sentences across concepts, without replacement or global RNG changes."""
    with open(csv_path, newline="", encoding="utf-8") as stream:
        sentences = list(dict.fromkeys(row["sentence"] for row in csv.DictReader(stream)))
    if not 2 <= sample_size <= len(sentences):
        raise ValueError(
            f"DMD needs 2 <= sample_size <= {len(sentences)} unique sentences; "
            f"requested {sample_size}. Provide a larger corpus for more samples."
        )
    return random.Random(seed).sample(sentences, sample_size)


def collect_final_token_states(model, tokenizer, prompts, *, batch_size=8, max_ctx=32):
    """Return CPU float32 states [layers + 1, prompts, hidden], before final norm.

    Reuse ContrastiveBuilder's final-token hooks, but retain individual samples
    instead of their mean. A one-token generation only observes the final valid
    position after max_ctx truncation; no generated-token states are fitted.
    """
    if not prompts or batch_size <= 0 or max_ctx <= 0:
        raise ValueError("Provide prompts and positive batch_size/max_ctx.")
    if model.training:
        raise ValueError("Final-token collection requires model.eval().")
    collector = ContrastiveBuilder(model, tokenizer)
    states = torch.empty(
        collector.num_layers + 1, len(prompts), collector.hidden_size,
        device="cpu", dtype=torch.float32,
    )
    padding_side, pad_token = tokenizer.padding_side, tokenizer.pad_token
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or an EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        with tqdm(total=len(prompts), desc="DMD final-token states", unit="prompt") as progress:
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_ctx,
                ).to(model.get_input_embeddings().weight.device)
                mask = inputs["attention_mask"]
                if not torch.all(mask.any(dim=1)):
                    raise ValueError("Every prompt must contain at least one token.")
                positions = torch.arange(mask.shape[1], device=mask.device)
                collector._last_token_indices = (mask * positions).argmax(dim=1)
                collector._batch_activations = [None] * (collector.num_layers + 1)
                with collector, torch.inference_mode():
                    model.generate(
                        **inputs, max_new_tokens=1, do_sample=False, use_cache=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                if any(state is None for state in collector._batch_activations):
                    raise RuntimeError("DMD collection did not visit every decoder block.")
                for layer, state in enumerate(collector._batch_activations):
                    states[layer, start : start + len(batch)].copy_(state)
                progress.update(len(batch))
    finally:
        collector.remove_hooks()
        tokenizer.padding_side, tokenizer.pad_token = padding_side, pad_token
    return states


def fit_residual_dmd(states, *, rank=None, alpha=1e-5, standardize=False, std_floor=1e-5):
    """Fit one raw A=I+B per block using kooplearn 2.0.3 ridge/PCR.

    rank=None retains all H components and requires N > H centered samples.
    An integer rank requests an explicit truncation; invalid ranks never clamp.
    Use feature-space Ridge when N >= H to avoid an unnecessarily large N x N
    kernel eigendecomposition; otherwise use its linear-kernel equivalent.
    Returns float32 A, means/scales, retained ranks and resolved covariance ranks.
    The latter counts retained input covariance eigenvalues above
    max(N, H) * float64_eps * max(largest_eigenvalue, alpha), not ridge rank.
    Full model rank does not imply full data rank or a full-rank residual B.
    Fitting is CPU float64; collection stays in model dtype and Riccati uses FP32.

    kooplearn splits a trajectory at lag_time. Concatenating [X, residual] with
    lag_time=N creates exactly N matched pairs, not transitions between prompts.
    Ridge.estimator_.T is the column-vector residual map. Kernel U_/V_ factors
    give B.T = X.T @ U_ @ V_.T @ residual / N. We reconstruct explicitly because
    state predict() in 2.0.3 uses input snapshots as its reconstruction observable.
    """
    if states.ndim != 3 or states.shape[0] < 2 or states.shape[1] < 2 or states.shape[2] < 1:
        raise ValueError("states must have shape (layers + 1, samples >= 2, hidden).")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
        raise ValueError("rank must be None (full) or a positive integer.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive.")
    states = states.detach().to(device="cpu", dtype=torch.float32)
    num_layers, num_samples, hidden = states.shape[0] - 1, states.shape[1], states.shape[2]
    components = hidden if rank is None else rank
    if components > min(num_samples - 1, hidden):
        raise ValueError(
            f"Requested {components} components, but {num_samples} centered samples "
            f"in {hidden} dimensions support at most {min(num_samples - 1, hidden)}. "
            "Full-rank fitting requires more unique samples than hidden dimensions."
        )
    dynamics = torch.empty(num_layers, hidden, hidden, dtype=torch.float32)
    means = torch.empty(num_layers + 1, hidden, dtype=torch.float32)
    scales = torch.ones(num_layers, hidden, dtype=torch.float32)
    ranks = torch.empty(num_layers, dtype=torch.int64)
    data_ranks = torch.empty(num_layers, dtype=torch.int64)
    for layer in tqdm(range(num_layers), desc="Fitting residual DMD", unit="layer"):
        x = states[layer].numpy().astype(np.float64)
        residual = states[layer + 1].numpy().astype(np.float64) - x
        if not np.isfinite(x).all() or not np.isfinite(residual).all():
            raise ValueError(f"DMD states must be finite (layer {layer}).")
        x_mean, residual_mean = x.mean(axis=0), residual.mean(axis=0)
        means[layer].copy_(torch.from_numpy(x_mean))
        means[layer + 1].copy_(torch.from_numpy(x_mean + residual_mean))
        x -= x_mean
        residual -= residual_mean
        scale = np.maximum(x.std(axis=0), std_floor) if standardize else np.ones(hidden)
        x /= scale
        residual /= scale
        scales[layer].copy_(torch.from_numpy(scale))
        paired_states = np.concatenate((x, residual), axis=0)
        if num_samples >= hidden:
            estimator = Ridge(
                reduced_rank=False, n_components=components, alpha=alpha,
                lag_time=num_samples, eigen_solver="dense", copy_X=False,
            ).fit(paired_states)
            raw_operator = estimator.estimator_.T.copy()
            if standardize:
                raw_operator *= scale[:, None]
                raw_operator /= scale[None, :]
        else:
            estimator = KernelRidge(
                kernel="linear", reduced_rank=False, n_components=components,
                alpha=alpha, lag_time=num_samples, eigen_solver="dense", copy_X=False,
            ).fit(paired_states)
            input_factor = (x.T @ estimator.U_) / math.sqrt(num_samples)
            output_factor = (residual.T @ estimator.V_) / math.sqrt(num_samples)
            input_factor /= scale[:, None]
            output_factor *= scale[:, None]
            raw_operator = output_factor @ input_factor.T
        if estimator.rank_ != components:
            raise ValueError(
                f"Layer {layer}: kooplearn retained {estimator.rank_}/{components} "
                "components. Increase alpha rather than silently truncating."
            )
        # PCR U columns are eigenvectors / sqrt(covariance_eigenvalue + alpha).
        covariance_eigenvalues = np.maximum(1 / np.square(estimator.U_).sum(axis=0) - alpha, 0)
        tolerance = max(num_samples, hidden) * np.finfo(np.float64).eps * max(covariance_eigenvalues.max(), alpha)
        data_ranks[layer] = int(np.count_nonzero(covariance_eigenvalues > tolerance))
        raw_operator.flat[:: hidden + 1] += 1.0
        dynamics[layer].copy_(torch.from_numpy(raw_operator))
        ranks[layer] = estimator.rank_
        if not torch.isfinite(dynamics[layer]).all():
            raise ValueError(f"Non-finite DMD operator at layer {layer}.")
        del estimator, paired_states
    return {"A": dynamics, "mean": means, "scale": scales, "rank": ranks, "data_rank": data_ranks}


def load_or_fit_residual_dmd(
    model, tokenizer, cache_dir, *, sample_size=None, rank=None, alpha=1e-5,
    standardize=False, std_floor=1e-5, batch_size=8, max_ctx=32, seed=0,
):
    """Reuse notebook-compatible mixed-corpus state/fit caches; return (fit, path).

    The default uses 2*H unique sentences and all H components, independently of
    the concept-specific steering direction. Reusing a fit requires no forwards;
    changing rank, ridge, or fitting scale reuses the collected final-token states.
    """
    hidden = model.get_input_embeddings().weight.shape[1]
    num_layers = len(model.model.layers)
    sample_size = 2 * hidden if sample_size is None else sample_size
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 2:
        raise ValueError("DMD sample_size must be an integer >= 2.")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0):
        raise ValueError("DMD rank must be None (full) or a positive integer.")
    components = hidden if rank is None else rank
    if components > min(sample_size - 1, hidden):
        raise ValueError(
            f"Requested {components} DMD components, but {sample_size} centered samples "
            f"in {hidden} dimensions support at most {min(sample_size - 1, hidden)}. "
            "Full-rank fitting requires more samples than hidden dimensions."
        )
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("DMD alpha must be finite and positive.")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("DMD std_floor must be finite and positive.")
    if batch_size <= 0 or max_ctx <= 0:
        raise ValueError("DMD batch_size and max_ctx must be positive.")
    dtype_tag = str(next(model.parameters()).dtype).removeprefix("torch.")
    state_tag = f"mixed_n{sample_size}_ctx{max_ctx}_seed{seed}_{dtype_tag}"
    rank_tag = "full" if rank is None else str(rank)
    mode = "standardized_fit" if standardize else "centered"
    fit_tag = f"{state_tag}_rank{rank_tag}_alpha{alpha}_{mode}_floor{std_floor}"
    cache_dir = Path(cache_dir)
    state_path = cache_dir / f"states_dmd_{state_tag}.pt"
    fit_path = cache_dir / f"A_dmd_{fit_tag}.pt"
    if fit_path.exists():
        tqdm.write(f"Loading DMD fit: {fit_path.name}")
        return torch.load(fit_path, map_location="cpu", weights_only=True), fit_path
    if state_path.exists():
        tqdm.write(f"Loading DMD states: {state_path.name}")
        states = torch.load(state_path, map_location="cpu", weights_only=True)
    else:
        prompts = load_dmd_prompts(sample_size, seed=seed)
        states = collect_final_token_states(
            model, tokenizer, prompts, batch_size=batch_size, max_ctx=max_ctx,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(states, state_path)
    expected_shape = (num_layers + 1, sample_size, hidden)
    if states.shape != expected_shape:
        raise ValueError(f"DMD states must have shape {expected_shape}; got {tuple(states.shape)}.")
    fit = fit_residual_dmd(
        states, rank=rank, alpha=alpha, standardize=standardize, std_floor=std_floor,
    )
    torch.save(fit, fit_path)
    tqdm.write(f"Saved DMD fit: {fit_path.name}")
    return fit, fit_path
