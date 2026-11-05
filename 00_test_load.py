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
import sys

REPO_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents)
     if (path / "llm" / "model.py").is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Open this notebook from the repository or one of its subdirectories.")
sys.path.insert(0, str(REPO_ROOT))

from llm.model import load_model
from llm.steering import sample
from llm.judge import build_judge, COHERENCE

# %%
device = "cuda"
dtype = "auto"  # Keep each model's native dtype (TinyStories: float32; Qwen: bfloat16).

# %%
model_name = "roneneldan/TinyStories-28M"
# model_name = "Qwen/Qwen3-4B"
tokenizer, model = load_model(model_name, device_map=device, dtype=dtype)

# %%
# Use a raw story prefix with either model, rather than a chat template.
prompt = "Once upon a time"
generated_ids = sample(model, tokenizer, prompt, n=1, max_new_tokens=128)
prompt_length = len(tokenizer(prompt)["input_ids"])
generated_output = tokenizer.decode(
    generated_ids[0, prompt_length:], skip_special_tokens=True
)
print("Prompt:", prompt)
print("Continuation:", generated_output)

# %%
judge = build_judge(device_map=device)
meets_criterion = judge(COHERENCE, prompt, [generated_output])[0]
print("Coherent:", meets_criterion)
