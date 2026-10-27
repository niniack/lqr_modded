#!/usr/bin/env bash
# Gemma-2-2B toxicity reproduction using the released steering/evaluation functions.
# Upstream has no Gemma-2-2B S-PID entry point at Table-1 scale, so this calls its run_trials with the
# released Gemma-2-2B settings from steer/tox_evalPID.py (kp=0.7, ki=0.01, kd=0.1, lambda in {0.5, 1})
# and the same protocol as the A-LQR runner: 5 x 1000 random RTP prompts.
# Needs ~/pickles/gemma-2-2b-{tox,nontox}.pkl (steer/toxicity/tox_data_script.py --model gemma2b).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p steer/toxicity/tox_data

TRIALS="${TRIALS:-1000}" SETS="${SETS:-5}" uv run --locked --group baseline python -u -c '
import json, os
import steer.toxicity.tox_data_script as utils
from steer.toxicity.test_toxicityPID import PATH, load_file, run_trials
from steer.benchmarks.ppl_from_file import get_ppl_from_file

name = "gemma-2-2b-spid-table1"
model, tokenizer = utils.load_model("google/gemma-2-2b", quant=True)
X_contr = load_file("gemma-2-2b-nontox")["X"] - load_file("gemma-2-2b-tox")["X"]
prompts = utils.get_tox_prompts(0.0, 1)

sweeps = []
for _ in range(int(os.environ["SETS"])):
    sweeps += run_trials(model, tokenizer, prompts, int(os.environ["TRIALS"]), X_contr,
                         kp=0.7, ki=0.01, kd=0.1, l_list=[0.5, 1], filename=name)
with open(PATH / (name + ".txt"), "w", encoding="utf-8") as f:
    json.dump(sweeps, f, indent=4)

del model
get_ppl_from_file(name, path=PATH)
'

# Five-shot MMLU, 1000 questions per lambda, using the same steering parameters.
MMLU_LOOPS="${MMLU_LOOPS:-100}" uv run --locked --group baseline python -u -c '
import json, os
from steer.toxicity.test_toxicityPID import PATH, load_file
from steer.toxicity.tox_data_script import load_model
from steer.benchmarks.testMMLU import test_mmlu_PID

model, tokenizer = load_model("google/gemma-2-2b", quant=True)
E = load_file("gemma-2-2b-nontox")["X"] - load_file("gemma-2-2b-tox")["X"]
scores = test_mmlu_PID(model, tokenizer, E, kp=0.7, ki=0.01, kd=0.1,
                       lambda_list=[0.5, 1], N_PROMPTS=10,
                       N_LOOP=int(os.environ["MMLU_LOOPS"]), N_SHOTS=5)
(PATH / "gemma-2-2b-spid-mmlu.json").write_text(json.dumps(scores, indent=2) + "\n")
'
