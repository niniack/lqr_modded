#!/usr/bin/env bash
# Gemma-2-2B toxicity reproduction using the released steering/evaluation functions.
# steer/toxicity/test_toxicity.py main(): 5 x 1000 random RTP prompts, lambda=3.5, q=0.1, r=10, qf=1.
# Needs ~/pickles/gemma-2-2b-{tox,nontox,nontox_jac}.pkl (steer/toxicity/tox_data_script.py --model gemma2b).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p steer/toxicity/tox_data

# Generation + classifier toxicity + Dist-n. One CSV line per set:
# lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,...,dist3_steered
uv run --locked --group baseline python -u -m steer.toxicity.test_toxicity

# Mistral-7B perplexity of the saved generations (the script's output name is the upstream one).
uv run --locked --group baseline python -u -c '
from steer.benchmarks.ppl_from_file import get_ppl_from_file
from steer.toxicity.test_toxicity import PATH
get_ppl_from_file("gemma-2-9b-tox-prelim", path=PATH)
'

# Five-shot MMLU, 1000 questions, using the same steering parameters.
uv run --locked --group baseline python -u -c '
import json
from steer.toxicity.test_toxicity import PATH, load_file
from steer.toxicity.tox_data_script import load_model
from steer.benchmarks.testMMLU import test_mmlu

model, tokenizer = load_model("google/gemma-2-2b", quant=True)
E = load_file("gemma-2-2b-nontox")["X"] - load_file("gemma-2-2b-tox")["X"]
A = load_file("gemma-2-2b-nontox_jac")["A"]
scores = test_mmlu(model, tokenizer, E, A, lambda_list=[3.5], q=0.1, r=10, qf=1,
                   N_PROMPTS=10, N_LOOP=100, N_SHOTS=5)
(PATH / "gemma-2-2b-alqr-mmlu.json").write_text(json.dumps(scores, indent=2) + "\n")
'
