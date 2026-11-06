from __future__ import annotations

import csv
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SENTENCES_CSV = Path(__file__).resolve().parent.parent / "data" / "filtered_sentences.csv"


def load_model(name, dtype="auto", device_map="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map).eval()
    return tokenizer, model


def position_ids(mask):
    """Positions for left-padded batches, as model.generate builds them."""
    positions = mask.long().cumsum(-1) - 1
    return positions.masked_fill_(mask == 0, 1)


class LayerInputs:
    """Capture block inputs and the final pre-norm state, including active pushes."""

    def __init__(self, model):
        self.modules = [*model.model.layers, model.model.norm]
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
    """(L + 1, prompts, hidden) float32 CPU states at each prompt's last token."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    states = []
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Final-token states"):
            batch = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                              truncation=max_ctx is not None, max_length=max_ctx).to(model.device)
            with LayerInputs(model) as record:
                model.model(**batch, position_ids=position_ids(batch["attention_mask"]), use_cache=False)
            states.append(torch.stack([state[:, -1].float().cpu() for state in record.states]))
    finally:
        tokenizer.padding_side = side
    return torch.cat(states, dim=1)


def load_concept_prompts(concept, sample_size=200, csv_path=SENTENCES_CSV):
    """Sample target sentences and a background balanced across the other concepts."""
    target, others = [], defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["concept"] == f"{concept}.NOUN":
                target.append(row["sentence"])
            else:
                others[row["concept"]].append(row["sentence"])
    per_concept = min(len(sentences) for sentences in others.values())
    background = [others[label][index] for index in range(per_concept) for label in others]
    return random.sample(target, sample_size), random.sample(background, sample_size)


def concept_direction(model, tokenizer, concept, sample_size=200, batch_size=8):
    """E (L + 1, hidden): mean last-token state of target minus background sentences."""
    target, background = load_concept_prompts(concept, sample_size)
    target_states = final_token_states(model, tokenizer, target, batch_size=batch_size)
    background_states = final_token_states(model, tokenizer, background, batch_size=batch_size)
    return target_states.mean(dim=1) - background_states.mean(dim=1)
