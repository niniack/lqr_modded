from __future__ import annotations

import copy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm.auto import tqdm

from .model import LayerInputs, blocks

_LAYER_KWARGS = ("attention_mask", "position_ids", "position_embeddings")


def _to(value, device):
    if isinstance(value, tuple):
        return tuple(_to(item, device) for item in value)
    if isinstance(value, torch.Tensor):
        return value.to(device, torch.float32) if value.is_floating_point() else value.to(device)
    return value


def efficient_mean_jacobian(model, tokenizer, prompts, *, max_ctx=32, chunk_size=128):
    """(L, H, H) float32 CPU mean of last-token block Jacobians over prompts.

    Other positions are held at their clean values. Blocks use float32, chunked
    jacrev, and math attention during differentiation; nominal model forwards keep
    their normal attention backend. chunk_size batches output derivatives, not prompts.
    """
    layers = blocks(model)
    recorded = []
    for prompt in tqdm(prompts, desc="Prompt states"):
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(model.device)
        kwargs = {}

        def grab_kwargs(module, args, layer_kwargs):
            kwargs.update({name: layer_kwargs[name] for name in _LAYER_KWARGS if layer_kwargs.get(name) is not None})

        grab = layers[0].register_forward_pre_hook(grab_kwargs, with_kwargs=True)
        try:
            with LayerInputs(model) as record, torch.no_grad():
                model.base_model(**ids, use_cache=False)
        finally:
            grab.remove()
        recorded.append(([state[0].float().cpu() for state in record.states[:-1]], kwargs))

    hidden = model.get_input_embeddings().weight.shape[1]
    total = torch.zeros(len(layers), hidden, hidden)
    with tqdm(total=len(layers) * len(recorded), desc="Efficient mean Jacobian", unit="Jacobian") as progress:
        for index, layer in enumerate(layers):
            block = copy.deepcopy(layer).float().eval().requires_grad_(False)
            device = next(block.parameters()).device
            with sdpa_kernel(SDPBackend.MATH):
                for states, kwargs in recorded:
                    x = states[index].to(device)
                    block_kwargs = {name: _to(value, device) for name, value in kwargs.items()}

                    def last_output(last):
                        out = block(torch.cat((x[:-1], last[None]))[None], **block_kwargs)
                        return (out[0] if isinstance(out, tuple) else out)[0, -1]

                    total[index] += torch.func.jacrev(last_output, chunk_size=chunk_size)(x[-1]).detach().cpu()
                    progress.update(1)
            del block
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return total / len(recorded)
