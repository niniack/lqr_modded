from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from .model import final_token_states

SENTENCES_CSV = Path(__file__).resolve().parent.parent / "data" / "filtered_sentences.csv"


def _rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_concept_prompts(concept, sample_size=200, csv_path=SENTENCES_CSV):
    """Sample target sentences and a background balanced across the other concepts."""
    target, others = [], defaultdict(list)
    for row in _rows(csv_path):
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
