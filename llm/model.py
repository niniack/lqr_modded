from __future__ import annotations

from functools import partial

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, dtype="auto", device_map="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map).eval()
    return tokenizer, model


def _stack(model):
    """(decoder blocks, final norm). model.base_model is the stack under the LM head."""
    base = model.base_model
    if hasattr(base, "layers") and hasattr(base, "norm"):  # Qwen, Llama
        return base.layers, base.norm
    if hasattr(base, "h") and hasattr(base, "ln_f"):  # GPT-Neo (TinyStories), GPT-2
        return base.h, base.ln_f
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def blocks(model):
    return _stack(model)[0]


def final_norm(model):
    return _stack(model)[1]


def position_ids(mask):
    """Positions for left-padded batches, as model.generate builds them."""
    positions = mask.long().cumsum(-1) - 1
    return positions.masked_fill_(mask == 0, 1)


class LayerInputs:
    """Record the hidden states entering each decoder block and the final norm.

    states[l] is (batch, seq, hidden) for l < L; states[L] is the last block's
    output before normalization. Steering pushes are included when active.
    """

    def __init__(self, model):
        self.modules = [*blocks(model), final_norm(model)]
        self.states = [None] * len(self.modules)
        self.handles = []

    def _save(self, index, module, args, kwargs):
        self.states[index] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    def __enter__(self):
        self.handles = [module.register_forward_pre_hook(partial(self._save, index), with_kwargs=True)
                        for index, module in enumerate(self.modules)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def final_token_states(model, tokenizer, prompts, *, batch_size=8, max_ctx=None):
    """(L + 1, prompts, hidden) float32 on CPU: each prompt's last token entering every block."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    states = []
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Final-token states"):
            batch = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                              truncation=max_ctx is not None, max_length=max_ctx).to(model.device)
            with LayerInputs(model) as record:
                model.base_model(**batch, position_ids=position_ids(batch["attention_mask"]), use_cache=False)
            states.append(torch.stack([state[:, -1].float().cpu() for state in record.states]))
    finally:
        tokenizer.padding_side = side
    return torch.cat(states, dim=1)
