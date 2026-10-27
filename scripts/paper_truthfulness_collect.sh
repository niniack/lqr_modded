#!/usr/bin/env bash
# Collect canonical TruthfulQA assets with the released ContrastiveBuilder methods.
# Counts are explicit reproduction settings, not claimed to be the paper's unpublished calibration counts.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --frozen --group baseline python -u -c '
import json, random
import numpy as np
import torch
from steer.data_handling import ContrastiveBuilder, PICKLE_JAR
from steer.truthfulness.tqa_data_script import load_model, tqa_prompts

SEED = 42
N_TRUE, N_FALSE, N_JACOBIANS = 200, 200, 50
MAX_JACOBIAN_CONTEXT = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
PICKLE_JAR.mkdir(parents=True, exist_ok=True)
paths = [PICKLE_JAR / ("gemma-2-2b-" + suffix + ".pkl")
         for suffix in ["true", "false", "true_jac"]]
if any(path.exists() for path in paths):
    raise FileExistsError("TruthfulQA assets already exist; move them aside before recollecting.")

true_pool, false_pool = tqa_prompts()
true_prompts = random.sample(true_pool, N_TRUE)
false_prompts = random.sample(false_pool, N_FALSE)
jacobian_prompts = random.sample(true_pool, N_JACOBIANS)
manifest = {
    "model": "google/gemma-2-2b", "seed": SEED,
    "true_prompts": true_prompts, "false_prompts": false_prompts,
    "jacobian_prompts": jacobian_prompts,
    "max_jacobian_context": MAX_JACOBIAN_CONTEXT,
    "note": "Released collection functions; 200/200/50 are explicit reproduction settings."
}
(PICKLE_JAR / "gemma-2-2b-truthfulness-calibration.json").write_text(json.dumps(manifest, indent=2) + "\n")
model, tokenizer = load_model("google/gemma-2-2b", quant=True)
collector = ContrastiveBuilder(model, tokenizer)
print("Collecting truthful means", flush=True)
collector.collect_data_batch(true_prompts, N_TRUE, "gemma-2-2b-true")
print("Collecting false means", flush=True)
collector.collect_data_batch(false_prompts, N_FALSE, "gemma-2-2b-false")
print("Collecting truthful-prompt Jacobians", flush=True)
collector.collect_jacobians(jacobian_prompts, N_JACOBIANS, "gemma-2-2b-true_jac",
                            max_ctx=MAX_JACOBIAN_CONTEXT)
print("TruthfulQA calibration complete:", PICKLE_JAR.resolve(), flush=True)
'
