# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: lqr-activation-steering (3.11.14)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import random
import sys

import torch

REPO_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents)
     if (path / 'llm' / 'model.py').is_file()
     and (path / 'data' / 'filtered_sentences.csv').is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError('Open this notebook from the repository or one of its subdirectories.')
sys.path.insert(0, str(REPO_ROOT))

from llm.concepts import concept_direction
from llm.model import blocks, load_model

# %%
MODEL_NAME = 'roneneldan/TinyStories-28M'
# MODEL_NAME = 'Qwen/Qwen3-4B'
CONCEPT = 'football'  # CSV labels are concept names such as 'football.NOUN'
SAMPLE_SIZE = 200
BATCH_SIZE = 32
SEED = 0
MAX_NEW_TOKENS = 96
STEERING_STRENGTH = 6.0  # Try 0.5, 2.0, etc.

DTYPE = 'auto'
DEVICE_MAP = 'auto'

# %%
tokenizer, model = load_model(
    MODEL_NAME,
    dtype=DTYPE,
    device_map=DEVICE_MAP,
)
model.eval()
print(f'Loaded {MODEL_NAME} on {model.device}')

# %%
random.seed(SEED)
torch.manual_seed(SEED)
direction = concept_direction(
    model,
    tokenizer,
    CONCEPT,
    sample_size=SAMPLE_SIZE,
    batch_size=BATCH_SIZE,
)
print('Direction shape (embedding + each layer, hidden size):', tuple(direction.shape))

# %%
decoder_layers = blocks(model)
LAYER_INDEX = len(decoder_layers) // 2  # zero-based block index
DIRECTION_INDEX = LAYER_INDEX + 1  # row 0 is the embedding output
if direction.shape[0] != len(decoder_layers) + 1:
    raise ValueError('Direction rows do not match this model’s decoder layers.')
layer_direction = direction[DIRECTION_INDEX]
print(f'Steering at decoder block {LAYER_INDEX}')
print(f'Layer direction norm: {torch.linalg.vector_norm(layer_direction).item():.3f}')


# %% [markdown]
# Select TinyStories or uncomment Qwen in the settings cell, then run the notebook top to bottom. Both models receive the raw story prefix without a chat template. `concept_direction` returns the target-minus-background mean states: row 0 is the embedding output, and the direction row for block `i` is `direction[i + 1]` (before final normalization for the last block). The hook adds that vector to every token’s block output during prefill and generation. Change `LAYER_INDEX` or `STEERING_STRENGTH` to explore this deliberately simple baseline.

# %%
def generate_text(prompt, steering_strength=0.0):
    hook_handle = None
    if steering_strength != 0.0:
        def add_direction_to_block_output(module, module_inputs, module_output):
            if isinstance(module_output, (tuple, list)):
                hidden_states = module_output[0]
            elif isinstance(module_output, torch.Tensor):
                hidden_states = module_output
            else:
                raise TypeError(f'Unsupported decoder block output: {type(module_output)}')

            vector = layer_direction.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            steered_hidden_states = hidden_states + steering_strength * vector
            if isinstance(module_output, tuple):
                return (steered_hidden_states, *module_output[1:])
            if isinstance(module_output, list):
                return [steered_hidden_states, *module_output[1:]]
            return steered_hidden_states

        hook_handle = decoder_layers[LAYER_INDEX].register_forward_hook(
            add_direction_to_block_output
        )

    try:
        model_inputs = tokenizer([prompt], return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        new_tokens = generated_ids[0, model_inputs.input_ids.shape[-1]:].tolist()
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    finally:
        if hook_handle is not None:
            hook_handle.remove()


# %%
PROMPT = 'Once upon a time'

baseline = generate_text(PROMPT)
steered = generate_text(PROMPT, steering_strength=STEERING_STRENGTH)
print('BASELINE\n========\n' + baseline)
print(f'\nSTEERED (strength={STEERING_STRENGTH})\n==============================\n' + steered)
