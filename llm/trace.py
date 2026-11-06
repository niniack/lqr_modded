"""Steered vs unsteered replay of sampled completions, on the same tokens.

Generation pushes the last token at every step, so one teacher-forced pass that
pushes every position from the last prompt token onward gives the same states.
A second pass without pushes gives the unsteered reference. Position t of the
result is the forward that produced generated token t.
"""

from __future__ import annotations

import torch

from .model import LayerInputs


@torch.inference_mode()
def trace(model, controller, lmbda, sequences, prompt_len):
    """Returns steered and unsteered states (L + 1, B, T, H) and valid (B, T): steps up to the first EOS."""
    ids = sequences[:, :-1]
    start = prompt_len - 1
    with LayerInputs(model) as record, controller.active(lmbda, push_from=start):
        model.base_model(input_ids=ids, use_cache=False)
    steered = torch.stack([state[:, start:] for state in record.states])
    with LayerInputs(model) as record:
        model.base_model(input_ids=ids, use_cache=False)
    nominal = torch.stack([state[:, start:] for state in record.states])
    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=sequences.device)
    is_eos = torch.isin(sequences[:, prompt_len:], eos).long()
    valid = (is_eos.cumsum(dim=1) - is_eos) == 0
    return {"steered": steered, "nominal": nominal, "valid": valid, "controller": controller, "lmbda": lmbda}


def _token_mean(values, valid):
    keep = valid & values.isfinite()
    return torch.where(keep, values, 0).sum(dim=1) / keep.sum(dim=1)


def completion_metrics(tr):
    """Per completion, at the final (pre-norm) state:
    along_v: v^T dx; change: |dx| / |x_unsteered|; orthogonal: |dx off v| / |x_unsteered|;
    push_energy: sum_l |u_l|^2 / |x_l|^2. All are token means over valid steps.
    """
    valid = tr["valid"]
    v = tr["controller"].v[-1].to(tr["steered"].device)
    dx = tr["steered"][-1].float() - tr["nominal"][-1].float()
    reference = tr["nominal"][-1].float().norm(dim=-1)
    along = dx @ v
    off = dx - along[..., None] * v
    energy = torch.zeros(valid.shape, device=valid.device)
    controller = tr["controller"]
    controller.lmbda = tr["lmbda"]
    for layer in controller.steered:
        x = tr["steered"][layer].float()
        u = controller.push(layer, x)
        energy += u.square().sum(-1) / x.square().sum(-1)
    columns = {
        "along_v": _token_mean(along, valid),
        "change": _token_mean(dx.norm(dim=-1) / reference, valid),
        "orthogonal": _token_mean(off.norm(dim=-1) / reference, valid),
        "push_energy": _token_mean(energy, valid),
    }
    return [{name: values[i].item() for name, values in columns.items()} for i in range(len(valid))]
