"""Fit/load the steering-response DMD gain operator used by notebook 03.

The fit replays fixed Identity A-LQR pushes, not sampled completions. Its
low-rank factors represent A = I + left @ basis.T for each decoder block.
"""

import csv
from functools import partial
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import LayerInputs, SENTENCES_CSV, load_concept_prompts
from .steering import QwenLQRSteering

_EVERY = (1, 2)
_SCALES = (0.1, 0.3, 1.0)
_RANKS = (8, 32, 128, 512, 2048)
_STRENGTH = 4


@torch.no_grad()
def _capture(model, inputs, pushes=None):
    """Record block inputs/final pre-norm state with fixed last-token pushes."""
    handles = []

    def add(layer, module, args, kwargs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        hidden = hidden.clone()
        hidden[:, -1] += pushes[layer].to(hidden.dtype)
        return type(output)((hidden, *output[1:])) if isinstance(output, (tuple, list)) else hidden

    try:
        if pushes is not None:
            for layer, block in enumerate(model.model.layers):
                handles.append(block.register_forward_hook(partial(add, layer), with_kwargs=True))
        with LayerInputs(model) as record, sdpa_kernel(SDPBackend.MATH):
            model.model(**inputs, use_cache=False)
        return record.states
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _reference_pushes(model, inputs, controller):
    pushes = torch.zeros(len(model.model.layers), inputs['input_ids'].shape[0], controller.E.shape[-1],
                         device=model.device, dtype=torch.float32)
    original = controller.push

    def record(layer, x):
        u = original(layer, x)
        pushes[layer].copy_(u[:, -1])
        return u

    controller.push = record
    try:
        with controller.active(_STRENGTH), sdpa_kernel(SDPBackend.MATH):
            model.model(**inputs, use_cache=False)
        return pushes
    finally:
        del controller.push


def _last(states):
    return torch.stack([x[:, -1] for x in states])


def _off(x, v):
    return x - (x * v).sum(dim=-1, keepdim=True) * v


def _apply_dmd(x, factors):
    left, basis = factors
    return x + (x @ basis) @ left.T


def _split(tokenizer, concept, train_pool, seed, max_ctx):
    # Preserve the original direction/Jacobian exclusions and held-out test set.
    random.seed(0)
    _, background = load_concept_prompts(concept, 200)
    random.seed(1)
    jacobian, _ = load_concept_prompts(concept, 50)

    def encode(text):
        return tuple(tokenizer(text, truncation=True, max_length=max_ctx)['input_ids'])

    excluded = {encode(text) for text in jacobian}
    candidates = {}
    for text in background:
        tokens = encode(text)
        if tokens not in excluded:
            candidates.setdefault(tokens, text)
    test = random.Random(0).sample(list(candidates.items()), 100)
    excluded.update(tokens for tokens, _ in test)
    with SENTENCES_CSV.open(newline='', encoding='utf-8') as stream:
        pool = {}
        for row in csv.DictReader(stream):
            if row['concept'] != f'{concept}.NOUN':
                tokens = encode(row['sentence'])
                if tokens not in excluded:
                    pool.setdefault(tokens, row['sentence'])
    selected = random.Random(seed).sample(list(pool.items()), min(train_pool, len(pool)))
    if len(selected) < 2:
        raise ValueError('Steering DMD needs at least two eligible training-pool prompts')
    held_out = max(1, int(np.ceil(len(selected) * 0.1)))
    return {'fit_prompts': selected[held_out:], 'validation_prompts': selected[:held_out], 'test_prompts': test}


@torch.no_grad()
def _responses(model, prompts, controllers):
    layers, hidden = controllers[0].E.shape[0] - 1, controllers[0].E.shape[1]
    count = len(prompts) * len(controllers) * len(_SCALES)
    d = torch.empty(layers + 1, count, hidden, dtype=torch.float32)
    u = torch.empty(layers, count, hidden, dtype=torch.float32)
    index = 0
    for tokens, _ in prompts:
        inputs = {'input_ids': torch.tensor([tokens], device=model.device)}
        nominal = _last(_capture(model, inputs))[:, 0]
        for controller in controllers:
            pushes = _reference_pushes(model, inputs, controller)
            for scale in _SCALES:
                states = _capture(model, inputs, scale * pushes)
                d[:, index] = (_last(states)[:, 0] - nominal).cpu()
                u[:, index] = (scale * pushes[:, 0]).cpu()
                index += 1
    return d, u


@torch.no_grad()
def _fit_dmd(training, validation, E, device):
    train_d, train_u = training
    val_d, val_u = validation
    directions = E.to(device=device, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    factors, scores = [], []
    for layer in range(len(train_u)):
        X = train_d[layer].T.to(device)
        Y = (train_d[layer + 1] - train_u[layer]).T.to(device)
        keep = X.ne(0).any(dim=0)
        X, Y = X[:, keep], Y[:, keep]
        if X.shape[1]:
            U, singular, Vh = torch.linalg.svd(X, full_matrices=False, driver='gesvd' if X.is_cuda else None)
            rank = min(max(_RANKS), int((singular > singular[0] * max(X.shape) * torch.finfo(X.dtype).eps).sum()))
            left, basis = ((Y - X) @ Vh[:rank].T) / singular[:rank], U[:, :rank]
        else:
            left = basis = X.new_empty(X.shape[0], 0)
        factors.append((left.cpu(), basis.cpu()))
        actual, applied, incoming = val_d[layer + 1].to(device), val_u[layer].to(device), val_d[layer].to(device)
        denominator = _off(actual.double(), directions[layer + 1]).norm(dim=-1)
        layer_scores = []
        for rank in _RANKS:
            prediction = _apply_dmd(incoming, (left[:, :rank], basis[:, :rank])) + applied
            errors = _off(prediction.double() - actual.double(), directions[layer + 1]).norm(dim=-1)
            keep = denominator > 0
            layer_scores.append(torch.quantile(errors[keep] / denominator[keep], 0.5).item() if keep.any() else np.nan)
        scores.append(layer_scores)
    curve = np.nanmean(scores, axis=0)
    chosen = _RANKS[int(np.nanargmin(curve))]
    return {'name': 'steering_dmd', 'chosen_rank': chosen,
            'factors': [(left[:, :chosen].clone(), basis[:, :chosen].clone()) for left, basis in factors],
            'validation': [{'rank': rank, 'err_perp': float(score)} for rank, score in zip(_RANKS, curve)]}


def steering_dmd(path, model, tokenizer, E, concept, q=0.1, r=10, qf=0.1, *,
                 train_pool=1000, seed=2, max_ctx=32):
    """Return the unchanged saved dict, or fit and save a missing FP32 cache.

    Existing caches are loaded on CPU without model forwards or dtype changes.
    A cache miss requires a caller-supplied float32 model. Identity A-LQR uses
    the supplied costs, lambda=4, every=1/2, and replay scales=.1/.3/1. Rank is
    selected from 8/32/128/512/2048 by mean layerwise median validation error
    perpendicular to E; the fitted operator is I + left @ basis.T.

    train_pool is the maximum number of non-target CSV prompts, deduplicated
    after max_ctx-token truncation. seed shuffles this pool before reserving
    its first ceil(10%) for validation. Defaults reproduce the original fit:
    seed-0 balanced background supplies 100 seed-0 held-out test prompts;
    seed-1 target prompts (50) are excluded as Jacobian inputs; the remaining
    pool uses seed=2. Test prompts never enter fitting or rank selection.
    Cache metadata must agree with the requested model/concept/costs and split
    settings. Original caches lacking the newer train_pool field retain the
    original default of 1000; no existing cache is overwritten or recomputed.
    """
    path = Path(path)
    layers, hidden = len(model.model.layers), model.get_input_embeddings().weight.shape[1]
    if E.shape != (layers + 1, hidden):
        raise ValueError('Concept directions must have shape (decoder layers + 1, hidden size)')
    config = dict(model=model.config._name_or_path, concept=concept, dtype='float32',
                  attention='math', q=q, r=r, qf=qf, max_ctx=max_ctx,
                  every=list(_EVERY), scales=list(_SCALES), ranks=list(_RANKS),
                  seed=seed, train_pool=train_pool)
    config['lambda'] = _STRENGTH
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if cached.get('name') != 'steering_dmd' or len(cached.get('factors', ())) != layers:
            raise ValueError(f'Incompatible steering DMD cache: {path}')
        saved = cached.get('config', {})
        legacy_defaults = dict(attention='math', every=list(_EVERY), scales=list(_SCALES),
                               ranks=list(_RANKS), seed=2, train_pool=1000)
        legacy_defaults['lambda'] = _STRENGTH
        mismatches = [key for key, value in config.items()
                      if saved.get(key, legacy_defaults.get(key)) != value]
        if mismatches:
            raise ValueError(f'Incompatible steering DMD cache {path}: {", ".join(mismatches)}')
        for left, basis in cached['factors']:
            if (left.ndim != 2 or left.shape != basis.shape or left.shape[0] != hidden
                    or left.dtype != torch.float32 or basis.dtype != torch.float32):
                raise ValueError(f'Incompatible steering DMD factors: {path}')
        return cached
    if train_pool < 2 or max_ctx < 1:
        raise ValueError('train_pool must be at least 2 and max_ctx must be positive')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Fitting steering DMD requires a float32 model; load one explicitly for a cache miss')
    split = _split(tokenizer, concept, train_pool, seed, max_ctx)
    identity = torch.eye(hidden, dtype=torch.float32).expand(layers, -1, -1)
    controllers = [QwenLQRSteering(model, identity, E, q, r, qf, every=every) for every in _EVERY]
    fitted = _fit_dmd(_responses(model, split['fit_prompts'], controllers),
                      _responses(model, split['validation_prompts'], controllers), E, model.device)
    fitted['config'] = dict(config, **split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(fitted, stream)
    return fitted
