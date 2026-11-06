"""Fixed-push replay, nominal-state JVPs, steering DMD, and cached-token grading."""

from copy import copy
from functools import partial
from pathlib import Path
import csv
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache, DynamicCache

from .concepts import SENTENCES_CSV, load_concept_prompts
from .model import LayerInputs, blocks
from .steering import LQRSteering


def _clone(value, layer):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Cache):
        if value.offloading:
            raise ValueError("JVP capture requires a non-offloaded cache")
        result = copy(value)
        result.layers = list(value.layers)
        if layer < len(value.layers):
            result.layers[layer] = copy(value.layers[layer])
            for name, item in vars(value.layers[layer]).items():
                setattr(result.layers[layer], name, _clone(item, layer))
        return result
    if isinstance(value, dict):
        return {key: _clone(item, layer) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item, layer) for item in value)
    return value


@torch.no_grad()
def capture(model, inputs, pushes=None, linearize=False):
    """Capture block inputs/final pre-norm state, optionally replaying fixed last-token pushes."""
    contexts, handles = [None] * len(blocks(model)), []

    def snapshot(layer, module, args, kwargs):
        with torch.inference_mode(False):
            contexts[layer] = (_clone(args, layer), _clone(kwargs, layer))

    def add(layer, module, args, kwargs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        hidden = hidden.clone()
        hidden[:, -1] += pushes[layer].to(hidden.dtype)
        return type(output)((hidden, *output[1:])) if isinstance(output, (tuple, list)) else hidden

    try:
        for layer, block in enumerate(blocks(model)):
            if linearize:
                handles.append(block.register_forward_pre_hook(partial(snapshot, layer), with_kwargs=True))
            if pushes is not None:
                handles.append(block.register_forward_hook(partial(add, layer), with_kwargs=True))
        with LayerInputs(model) as record, sdpa_kernel(SDPBackend.MATH):
            model.base_model(**{'use_cache': False, **inputs})
        return record.states, contexts
    finally:
        for handle in handles:
            handle.remove()


def jvp(block, context, direction, layer):
    """Differentiate only the last position; copied pre-forward KV tensors remain constant."""
    with torch.inference_mode(False), torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        args, kwargs = _clone(context, layer)
        hidden = args[0] if args else kwargs['hidden_states']

        def output(last):
            x = torch.cat((hidden[:, :-1], last[:, None]), dim=1)
            call_args, call_kwargs = args, dict(kwargs)
            if args:
                call_args = (x, *args[1:])
            else:
                call_kwargs['hidden_states'] = x
            result = block.forward(*call_args, **call_kwargs)
            return (result[0] if isinstance(result, (tuple, list)) else result)[:, -1]

        return torch.func.jvp(output, (hidden[:, -1],), (direction.detach().to(hidden).clone(),))[1]


@torch.no_grad()
def reference(model, inputs, controller, strength=4):
    pushes = torch.zeros(len(blocks(model)), inputs['input_ids'].shape[0], controller.v.shape[-1],
                         device=model.device)
    original = controller.push

    def record(layer, x):
        u = original(layer, x)
        pushes[layer].copy_(u[:, -1])
        return u

    controller.push = record
    try:
        with controller.active(strength):
            states, _ = capture(model, inputs)
        return states, pushes
    finally:
        del controller.push


def _last(states):
    return torch.stack([x[:, -1] for x in states])


def _off(x, v):
    return x - (x * v).sum(dim=-1, keepdim=True) * v


def _directions(E, device):
    v = E.to(device=device, dtype=torch.float64)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def apply_dmd(x, factors):
    left, basis = factors
    return x + (x @ basis) @ left.T


@torch.no_grad()
def first_token(model, E, A_jac, cache, scales=(0.01, 0.1, 0.3, 1.0), q=0.1, r=10, qf=0.1, strength=4):
    layers, device = list(blocks(model)), model.device
    identity = torch.eye(E.shape[-1]).expand(len(layers), -1, -1)
    controller = LQRSteering(model, identity, E, q, r, qf)
    v = _directions(E, device)[-1]
    factors = [tuple(x.to(device) for x in pair) for pair in cache['factors']]
    A_jac, rows = A_jac.to(device), []
    for prompt_id, (tokens, text) in enumerate(cache['config']['test_prompts']):
        inputs = {'input_ids': torch.tensor([tokens], device=device)}
        nominal, contexts = capture(model, inputs, linearize=True)
        _, pushes = reference(model, inputs, controller, strength)
        estimates = {name: torch.zeros_like(pushes[0]) for name in
                     ('identity', 'mean_jacobian', 'steering_dmd', 'prompt_jacobian')}
        for layer, block in enumerate(layers):
            estimates['identity'] += pushes[layer]
            estimates['mean_jacobian'] = estimates['mean_jacobian'] @ A_jac[layer].T + pushes[layer]
            estimates['steering_dmd'] = apply_dmd(estimates['steering_dmd'], factors[layer]) + pushes[layer]
            estimates['prompt_jacobian'] = jvp(block, contexts[layer], estimates['prompt_jacobian'], layer) + pushes[layer]
        for scale in scales:
            steered, _ = capture(model, inputs, pushes * scale)
            truth = (steered[-1][:, -1] - nominal[-1][:, -1]).double()
            denominator = _off(truth, v).norm(dim=-1)
            for name, estimate in estimates.items():
                error = _off(scale * estimate.double() - truth, v).norm(dim=-1) / denominator
                rows.append(dict(prompt_id=prompt_id, prompt=text, scale=scale, predictor=name,
                                 rel_dev=(truth.norm() / nominal[-1][:, -1].double().norm()).item(),
                                 err_perp=error.item()))
    return pd.DataFrame(rows)


def _split(tokenizer, concept):
    random.seed(0)
    _, background = load_concept_prompts(concept, 200)
    random.seed(1)
    jacobian, _ = load_concept_prompts(concept, 50)
    encode = lambda text: tuple(tokenizer(text, truncation=True, max_length=32)['input_ids'])
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
    selected = random.Random(2).sample(list(pool.items()), min(1000, len(pool)))
    held_out = max(1, int(np.ceil(len(selected) * 0.1)))
    return {'fit_prompts': selected[held_out:], 'validation_prompts': selected[:held_out], 'test_prompts': test}


@torch.no_grad()
def _responses(model, prompts, controllers):
    layers, hidden = controllers[0].v.shape[0] - 1, controllers[0].v.shape[1]
    count = len(prompts) * len(controllers) * 3
    d, u = torch.empty(layers + 1, count, hidden), torch.empty(layers, count, hidden)
    index = 0
    for tokens, _ in prompts:
        inputs = {'input_ids': torch.tensor([tokens], device=model.device)}
        clean, _ = capture(model, inputs)
        nominal = _last(clean)[:, 0]
        for controller in controllers:
            _, pushes = reference(model, inputs, controller)
            for scale in (0.1, 0.3, 1.0):
                states, _ = capture(model, inputs, scale * pushes)
                d[:, index] = (_last(states)[:, 0] - nominal).cpu()
                u[:, index] = (scale * pushes[:, 0]).cpu()
                index += 1
    return d, u


@torch.no_grad()
def fit_dmd(training, validation, E, device, ranks=(8, 32, 128, 512, 2048)):
    train_d, train_u = training
    val_d, val_u = validation
    directions = _directions(E, device)
    factors, scores = [], []
    for layer in range(len(train_u)):
        X = train_d[layer].T.to(device)
        Y = (train_d[layer + 1] - train_u[layer]).T.to(device)
        keep = X.ne(0).any(dim=0)
        X, Y = X[:, keep], Y[:, keep]
        if X.shape[1]:
            U, singular, Vh = torch.linalg.svd(X, full_matrices=False, driver='gesvd' if X.is_cuda else None)
            rank = min(max(ranks), int((singular > singular[0] * max(X.shape) * torch.finfo(X.dtype).eps).sum()))
            left, basis = ((Y - X) @ Vh[:rank].T) / singular[:rank], U[:, :rank]
        else:
            left = basis = X.new_empty(X.shape[0], 0)
        factors.append((left.cpu(), basis.cpu()))
        actual, applied, incoming = val_d[layer + 1].to(device), val_u[layer].to(device), val_d[layer].to(device)
        denominator = _off(actual.double(), directions[layer + 1]).norm(dim=-1)
        layer_scores = []
        for rank in ranks:
            prediction = apply_dmd(incoming, (left[:, :rank], basis[:, :rank])) + applied
            errors = _off(prediction.double() - actual.double(), directions[layer + 1]).norm(dim=-1)
            keep = denominator > 0
            layer_scores.append(torch.quantile(errors[keep] / denominator[keep], 0.5).item() if keep.any() else np.nan)
        scores.append(layer_scores)
    curve = np.nanmean(scores, axis=0)
    chosen = ranks[int(np.nanargmin(curve))]
    return {'name': 'steering_dmd', 'chosen_rank': chosen,
            'factors': [(left[:, :chosen].clone(), basis[:, :chosen].clone()) for left, basis in factors],
            'validation': [{'rank': rank, 'err_perp': float(score)} for rank, score in zip(ranks, curve)]}


def steering_dmd(path, model, tokenizer, E, concept, q=0.1, r=10, qf=0.1):
    path = Path(path)
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if cached['name'] != 'steering_dmd' or len(cached['factors']) != len(blocks(model)):
            raise ValueError(f'Incompatible steering DMD cache: {path}')
        return cached
    split = _split(tokenizer, concept)
    identity = torch.eye(E.shape[-1]).expand(len(blocks(model)), -1, -1)
    controllers = [LQRSteering(model, identity, E, q, r, qf, every=every) for every in (1, 2)]
    fitted = fit_dmd(_responses(model, split['fit_prompts'], controllers),
                     _responses(model, split['validation_prompts'], controllers), E, model.device)
    fitted['config'] = dict(model=model.config._name_or_path, concept=concept, dtype='float32',
                            q=q, r=r, qf=qf, max_ctx=32, **split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(fitted, stream)
    return fitted


@torch.no_grad()
def grade(model, nominal, E, A_jac, sequences, prompt_len, q=0.1, qf=0.1, strength=4):
    """Replay existing plain-A-LQR token IDs; both observers use the same actual pushes."""
    import json

    layers, device = list(blocks(nominal)), model.device
    v = _directions(E, device)[-1]
    A_jac = A_jac.to(device)
    identity = torch.eye(E.shape[-1]).expand(len(layers), -1, -1)
    rows = []
    for r, group in sequences.groupby('R', sort=True):
        group = group.sort_values('trial')
        ids = torch.tensor([json.loads(text) for text in group.token_ids], device=device)
        controller = LQRSteering(model, identity, E, q, float(r), qf)
        caches = [DynamicCache(config=m.config) for m in (model, nominal)]
        eos = model.generation_config.eos_token_id
        eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=device)
        stops = torch.isin(ids[:, prompt_len:], eos).long()
        valid = (stops.cumsum(1) - stops) == 0
        errors = []
        for step in range(ids.shape[1] - prompt_len):
            start = 0 if step == 0 else prompt_len + step - 1
            end = prompt_len + step
            inputs = dict(input_ids=ids[:, start:end], use_cache=True,
                          attention_mask=torch.ones(ids.shape[0], end, dtype=torch.long, device=device),
                          position_ids=torch.arange(start, end, device=device)[None].expand(ids.shape[0], -1))
            clean, contexts = capture(nominal, {**inputs, 'past_key_values': caches[1]}, linearize=True)
            actual, pushes = reference(model, {**inputs, 'past_key_values': caches[0]}, controller, strength)
            predicted = torch.zeros_like(pushes[0])
            shared = torch.zeros_like(predicted)
            for layer, block in enumerate(layers):
                predicted = jvp(block, contexts[layer], predicted, layer) + pushes[layer]
                shared = shared @ A_jac[layer].T + pushes[layer]
            truth = (actual[-1][:, -1] - clean[-1][:, -1]).double()
            denominator = _off(truth, v).norm(dim=-1)
            errors.append(torch.stack([_off(value.double() - truth, v).norm(dim=-1) / denominator
                                       for value in (predicted, shared)], dim=-1))
        errors = torch.stack(errors, dim=1)
        for index, row in enumerate(group.to_dict('records')):
            for col, predictor in enumerate(('prompt_jacobian', 'mean_jacobian')):
                row[f'{predictor}_error_step0'] = errors[index, 0, col].item()
                row[f'{predictor}_error_later'] = errors[index, 1:, col][valid[index, 1:]].mean().item()
            rows.append(row)
    return pd.DataFrame(rows)
