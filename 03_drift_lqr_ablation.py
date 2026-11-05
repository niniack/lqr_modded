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

# %% [markdown]
# # Drift-canceling A-LQR ablation
#
# Compare the current A-LQR push with orthogonal drift cancellation, using the same Identity, Mean Jacobian, and Random dynamics and sparse actuation settings as notebook 02:
#
# $$u_\ell = \alpha_\ell K_\ell v_\ell - w K_\ell d_{\perp,\ell},\qquad
# \alpha_\ell=\lambda\|E_\ell\|-v_\ell^\top x_\ell,$$
# $$d_\ell=x_\ell-x_\ell^{\mathrm{nominal}},\qquad
# d_{\perp,\ell}=d_\ell-(v_\ell^\top d_\ell)v_\ell.$$
#
# The independent, identical nominal model consumes **the same generated tokens**, not independently generated stories. `w=0` is the original controller; `w=1` adds cancellation. A full-sequence replay can differ slightly from cached generation in BF16, so replay metrics are approximate. Controller drift logs measure block **inputs**; `completion_metrics()['orthogonal']` measures the final **pre-norm output**, not those logs.
#
# The controller checks have already passed; this notebook now goes directly from loading E/A to the sweep. Experiment size is controlled by `PILOT`, `TRIALS`, `LAMBDAS`, `EVERY`, and `DRIFT_WEIGHTS` below. E/A caches are shared with notebook 02; results go to a separate drift directory, split by pilot/full and Q/R/QF.
#

# %%
from pathlib import Path
import gc, importlib, random, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

REPO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'llm' / 'steering.py').is_file())
sys.path.insert(0, str(REPO_ROOT))
import llm.model, llm.concepts, llm.jacobian, llm.steering, llm.trace, llm.judge
for module in (llm.model, llm.concepts, llm.jacobian, llm.steering, llm.trace, llm.judge):
    importlib.reload(module)
from llm.model import load_model, blocks
from llm.concepts import concept_direction, load_concept_prompts
from llm.jacobian import efficient_mean_jacobian
from llm.steering import DriftLQRSteering, sample
from llm.trace import trace, completion_metrics
from llm.judge import build_judge, concept_criterion, COHERENCE, repetition

MODEL_NAME = 'Qwen/Qwen3-4B'
# MODEL_NAME = 'Qwen/Qwen3-0.6B'
# MODEL_NAME = 'roneneldan/TinyStories-28M'
CONCEPT = 'football'
PROMPT = 'Once upon a time'
SEED = 0
DIRECTION_SAMPLES = 200
JACOBIAN_SAMPLES = 50
MAX_CTX = 32
BATCH_SIZE = 8
RANDOM_SEED = 0

Q, R, QF = 0.1, 10, 0.1
STEER_LAST_N_LAYERS = None
EVERY = (1, 2, 3)
DRIFT_WEIGHTS = (0.0, 0.5, 1.0)
PILOT = False
TRIALS = 20 if PILOT else 10
LAMBDAS = (2, 4) if PILOT else (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)
MAX_NEW_TOKENS = 96
TEMPERATURE, TOP_P = 0.7, 0.95
JUDGE_BATCH = 128


# %%
random.seed(SEED)
torch.manual_seed(SEED)
tokenizer, model = load_model(MODEL_NAME)
nominal = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=next(model.parameters()).dtype,
    device_map=str(model.device),
    attn_implementation=model.config._attn_implementation,
).eval()
NUM_LAYERS = len(blocks(model))
HIDDEN = model.get_input_embeddings().weight.shape[1]
PROMPT_LEN = len(tokenizer(PROMPT)['input_ids'])
ARTIFACTS = REPO_ROOT / 'artifacts' / MODEL_NAME.rsplit('/', 1)[-1].lower()
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RUN_DIR = ARTIFACTS / 'drift_lqr' / ('pilot' if PILOT else 'full') / f'q{Q:g}_r{R:g}_qf{QF:g}'
RUN_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_KWARGS = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
print(f'{MODEL_NAME}: {NUM_LAYERS} blocks, hidden {HIDDEN}, {next(model.parameters()).dtype}')
print('Nominal:', nominal.device, next(nominal.parameters()).dtype, nominal.config._attn_implementation)
print('Results:', RUN_DIR)


# %%
E_PATH = ARTIFACTS / f'E_{CONCEPT}_n{DIRECTION_SAMPLES}_seed{SEED}.pt'
if E_PATH.exists():
    E = torch.load(E_PATH, weights_only=True)
else:
    random.seed(SEED)
    E = concept_direction(model, tokenizer, CONCEPT, DIRECTION_SAMPLES, BATCH_SIZE)
    torch.save(E, E_PATH)
print('E:', tuple(E.shape))

# %%
A_PATH = ARTIFACTS / f'A_{CONCEPT}_n{JACOBIAN_SAMPLES}_ctx{MAX_CTX}_seed{SEED}.pt'
if A_PATH.exists():
    A_jac = torch.load(A_PATH, weights_only=True)
else:
    random.seed(SEED + 1)
    targets, _ = load_concept_prompts(CONCEPT, JACOBIAN_SAMPLES)
    A_jac = efficient_mean_jacobian(model, tokenizer, targets, max_ctx=MAX_CTX)
    torch.save(A_jac, A_PATH)

B_random = torch.randn(NUM_LAYERS, HIDDEN, HIDDEN, generator=torch.Generator().manual_seed(RANDOM_SEED))
size = torch.linalg.matrix_norm(A_jac - torch.eye(HIDDEN)) / torch.linalg.matrix_norm(B_random)
A_random = torch.eye(HIDDEN) + size[:, None, None] * B_random
del B_random

OPERATORS = {'Identity': torch.eye(HIDDEN).expand(NUM_LAYERS, -1, -1), 'Mean Jacobian': A_jac, 'Random': A_random}
print('Mean Jacobian:', A_PATH.name)

# %% [markdown]
# ## Sweep
#
# One controller per (operator, every) is reused across weights and then lambdas. Each weight/operator/sparsity uses matching seeds for a given lambda. The progress bar counts completion batches. The unsteered baseline is sampled separately without a controller and is labeled `every=0`, `drift_weight=0`; replay-only metrics are left missing for it, as in notebook 02.
#
# `along_v`, `change`, `orthogonal`, and `push_energy` are consumed immediately after each replay, before the controller changes. Unjudged completions are saved before loading Granite.
#

# %%
torch.manual_seed(SEED)
ids = sample(model, tokenizer, PROMPT, TRIALS, **SAMPLE_KWARGS)
results = [{'every': 0, 'operator': 'Unsteered', 'drift_weight': 0.0, 'lambda': 0.0,
            'completion': text.strip()}
           for text in tokenizer.batch_decode(ids[:, PROMPT_LEN:], skip_special_tokens=True)]
del ids

sweep_batches = len(EVERY) * len(OPERATORS) * len(DRIFT_WEIGHTS) * len(LAMBDAS)
with tqdm(total=sweep_batches, desc='Drift A-LQR sweep', unit='batch') as progress:
    for every in EVERY:
        for name, A in OPERATORS.items():
            controller = DriftLQRSteering(model, nominal, A, E, Q, R, QF,
                                          steer_last_n_layers=STEER_LAST_N_LAYERS, every=every)
            for weight in DRIFT_WEIGHTS:
                controller.drift_weight = weight
                for i, lmbda in enumerate(LAMBDAS):
                    torch.manual_seed(SEED + i)
                    ids = sample(model, tokenizer, PROMPT, TRIALS,
                                 controller=controller, lmbda=lmbda, **SAMPLE_KWARGS)
                    tr = trace(model, controller, lmbda, ids, PROMPT_LEN)
                    batch_metrics = completion_metrics(tr)
                    del tr
                    texts = tokenizer.batch_decode(ids[:, PROMPT_LEN:], skip_special_tokens=True)
                    for text, metrics in zip(texts, batch_metrics):
                        results.append({'every': every, 'operator': name, 'drift_weight': weight,
                                        'lambda': lmbda, **metrics, 'completion': text.strip()})
                    progress.set_postfix(every=every, operator=name, weight=weight, lmbda=lmbda)
                    progress.update(1)
                    del ids, batch_metrics, texts
            del controller
            gc.collect()
            torch.cuda.empty_cache()

results = pd.DataFrame(results)
results.to_csv(RUN_DIR / f'drift_ablation_{CONCEPT}_unjudged.csv', index=False)


# %% [markdown]
# ## Fast judge
#
# Free both language models before loading Granite Guardian. Its current fast yes/no API scores the concept and coherence separately with progress bars, deduplication, and length-sorted batches. `JUDGE_BATCH=128` matches the current notebook 02 setting; reduce it if GPU memory is limited. Judging can be rerun on the saved unjudged CSV without regenerating completions.
#

# %%
# All controllers and traces were released above; neither model remains referenced by them.
for name in ('model', 'nominal', 'judge'):
    globals().pop(name, None)
gc.collect()
torch.cuda.empty_cache()
judge = build_judge(batch_size=JUDGE_BATCH)
results = pd.read_csv(RUN_DIR / f'drift_ablation_{CONCEPT}_unjudged.csv')
texts = results['completion'].tolist()
print(f'Judging {len(texts):,} completions ({len(set(texts)):,} unique), two criteria, batch size {JUDGE_BATCH}.')
results['concept'] = judge(concept_criterion(CONCEPT), PROMPT, texts, desc=f'Concept: {CONCEPT} (1/2)')
results['coherent'] = judge(COHERENCE, PROMPT, texts, desc='Coherence (2/2)')
results['both'] = results['concept'] & results['coherent']
results['repetition'] = results['completion'].map(repetition)
results.to_csv(RUN_DIR / f'drift_ablation_{CONCEPT}.csv', index=False)


# %% [markdown]
# ## Coherence versus concept rate
#
# One figure per actuation interval; operator fixes the color and cancellation weight fixes the line style. Error bars on **both** axes are binomial standard errors $\sqrt{p(1-p)/n}$. Labels give lambda. The separately sampled unsteered baseline is a black star. These marginal error bars do not estimate uncertainty in paired differences.
#

# %%
groups = results.groupby(['every', 'operator', 'drift_weight', 'lambda'], sort=False)
summary = groups[['concept', 'coherent', 'both', 'repetition', 'along_v', 'change', 'orthogonal', 'push_energy']].mean()
summary['n'] = groups.size()
summary['concept_se'] = np.sqrt(summary['concept'] * (1 - summary['concept']) / summary['n'])
summary['coherent_se'] = np.sqrt(summary['coherent'] * (1 - summary['coherent']) / summary['n'])
summary.to_csv(RUN_DIR / f'drift_ablation_{CONCEPT}_summary.csv')
print(summary.round(3).to_string())

# %%
fig, axes = plt.subplots(len(EVERY), len(OPERATORS), figsize=(4 * len(OPERATORS), 3.6 * len(EVERY)),
                         sharex=True, sharey=True, squeeze=False)
for row, every in zip(axes, EVERY):
    for ax, name in zip(row, OPERATORS):
        block = summary.xs((every, name), level=['every', 'operator'])
        for weight, curve in block.groupby(level='drift_weight'):
            curve = curve.sort_index(level='lambda')
            ax.errorbar(curve['concept'], curve['coherent'], yerr=curve['coherent_se'], marker='o', capsize=2,
                        color='tab:blue' if weight == 0 else 'tab:red', label=f'w={weight:g}')
            for lmbda, point in zip(curve.index.get_level_values('lambda'), curve.itertuples()):
                ax.annotate(f'{lmbda:g}', (point.concept, point.coherent), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
        ax.plot(baseline['concept'], baseline['coherent'], 'k*', ms=9)
        ax.set_title(f'{name}, every={every}', fontsize=9)
        ax.grid(alpha=0.2)
axes[0, 0].set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
for ax in axes[-1]:
    ax.set_xlabel('Concept rate')
for ax in axes[:, 0]:
    ax.set_ylabel('Coherence rate')
axes[0, 0].legend(fontsize=8)
fig.suptitle(f'{MODEL_NAME}, {CONCEPT}: w=0 (blue) vs w=1 (red), labels = lambda')
fig.tight_layout()
fig.savefig(RUN_DIR / f'drift_ablation_{CONCEPT}_grid.png', dpi=150)
plt.show()
