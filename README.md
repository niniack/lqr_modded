# KAE Steering

Train a Koopman surrogate for a **Gemma suffix**, using streamed prompts and clean
teacher activations. The bundled recipe replaces Gemma 3 4B's last four blocks:
raw residual state after block 30 in, raw states after blocks 31–34 out.
Only the final nonpadding token is predicted. The frozen teacher still reads the
whole prompt using its native attention, masks, and positional embeddings.

## Start here

```bash
uv sync --locked
uv run --locked python -m kae.train --config configs/gemma3-4b.toml
```

Edit the one recipe in `configs/`. It contains the dataset, document budget,
model dimensions, loss weights, optimizer settings, and logging options. Fresh
runs require a new `output_dir`. Add `--no-wandb` to disable remote logging.

Dependencies are pinned, including PyTorch 2.9.1, Transformers 4.57.1, and
Datasets 4.4.2. Gemma requires an authorized Hugging Face login. Keep credentials
outside Git and TOML; use `uv run --env-file .env ...` if needed.

### Code map

| Location | Responsibility |
|---|---|
| `kae/train.py` | Small command-line entry point |
| `kae/models/` | AIKAE, invertible encoder/decoder, Koopman operators, optional context mixer |
| `kae/data/` | Prompt streaming, crops/shuffling, fixed validation data, tensor batches |
| `kae/data/cache.py` | Validation/test cache reader and 256-document shard writer |
| `kae/teacher.py` | Frozen native Gemma backbone and final normalization/vocabulary head |
| `kae/training/` | Training loop, objective assembly, optimization, validation, save/resume |
| `kae/losses.py` | Two plain functions: MSE and KL divergence |
| `kae/progress.py`, `kae/wandb.py` | Console progress, shared run metadata, and epoch-based W&B logging |
| `kae/checkpoints.py` | Student construction, strict weight loading, atomic checkpoint files |

To read the core, start with `models/aikae.py`, `training/fit.py`, and
`training/objectives.py`. There are no controller, perturbation, Jacobian,
generation-study, migration-tool, or pilot-launcher packages inside `kae/`.
The independent upstream baseline remains in `steer/`.

## Data and teacher

Training is **streaming-only**: one persistent shuffled source continues across
logical epochs. `docs_per_epoch` is a fresh accepted-document budget, not the
size of a replayed dataset. Teacher targets are transient; no training activation
cache or separate calibration/PCA pass is required.

Each document is tokenized and capped, checked against held-out token identities,
then cropped to a seeded random prefix between `min_truncation` and its capped
length. Very short documents are retained. Exclusion uses the capped unpadded
token sequence **before cropping**; this is not corpus-wide document deduplication.
The first `min(256, docs_per_epoch)` training documents also form a fixed training
probe. Their teacher targets are reused by the first epoch, not recomputed.

The validation cache fixes the teacher checkpoint, contiguous block boundaries,
maximum length, and held-out prompts. **Reuse `data/activations/gemma3-4b`** if it
is already present. Only create a new cache when changing those choices:

```bash
uv run --locked python -m kae.cache_activations \
  --model google/gemma-3-4b-pt \
  --model-revision cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --last-layers 4 --val-docs 512 --test-docs 512 \
  --max-length 512 --batch-size 2 --device cuda \
  --output-dir data/activations/gemma3-4b
```

Alternatively, `--layers START END` selects an inclusive contiguous range of
raw block outputs. Caches contain only validation/test data, in 256-document
shards with exact unpadded token IDs. Validation and test token identities are
disjoint. Nonempty output directories are refused, not resumed. Historical extra
shards in existing caches are ignored; stored user assets are not deleted.

Extraction runs the backbone, not the vocabulary head, and retains raw states
**before final normalization**. FP32 weights, suffix, head, and output states are
preserved. `teacher_precision = "bfloat16"` uses BF16 autocast only in the prefix
on supported CUDA GPUs; use `"float32"` on CPU or V100.

For another Gemma, create its fixed cache and change `validation_cache`. Omit
`latent_width` to use the teacher's hidden width without augmentation. Optional
per-boundary student normalization is fitted from the existing training probe;
it does not change the raw-state MSE definition. Loading Gemma's pretrained
weight shards is normal even when the KAE itself is trained from scratch.

For local data, use `dataset = "parquet"` or `"json"` with `data_files`; omit
`dataset_config` and `dataset_revision`. For Hub datasets, specify their actual
configuration and pin a revision as in the bundled recipe.

## Clean objectives

| Recipe weight | Meaning |
|---|---|
| `state_weight` | Raw MSE over every decoded output in the input-start rollout |
| `latent_weight` | MSE between propagated latents and encodings of the true states; both branches trainable |
| `kl_weight` | Final-endpoint KL(teacher ‖ student) through the frozen final norm and vocabulary head |
| `multistart_state_weight` | Additional raw state MSE from later teacher starting boundaries |
| `multistart_latent_weight` | Additional latent MSE from those same later starts |

Every rollout encodes its starting state once, then propagates without resetting
to teacher targets. For four replaced blocks, later starts contribute six
start/end pairs, each weighted equally. No additional teacher forwards are needed.

These remain **ordinary MSE losses**, not FVU-normalized objectives. Reconstruction
is diagnostic only. There are no perturbation/JVP targets, losses, or diagnostics.
The bundled recipe retains its previous state/KL weights; latent and multi-start
weights are explicit and initially zero. Zero-weight losses are still measured
on both fixed evaluation sets, so enabling an objective is a deliberate choice.

Optimization uses Muon for matrix weights and Adam for biases/non-matrix
parameters. `learning_rate` is the Adam base peak; `muon_lr_multiplier` explicitly
scales the matrix peak. Both use linear warmup followed by cosine decay. Combined
gradients are clipped at `grad_clip` before updates.

## Logging and checkpoints

W&B logs online to **tokae**. Model, dataset/configuration/revision, batch sizes,
document budgets, dimensions, precision, epochs, and every loss weight are baked
into the run configuration. No silent offline fallback is used.

- **One epoch axis:** epoch 0 is initialization; epochs 1–N are completed epochs.
- **Comparable populations:** `train/*` is the fixed training probe, evaluated at
  the same checkpoint as the fixed held-out `val/*` set.
- **Quality:** raw state/latent MSE and FVU, per-layer state metrics, KL,
  reconstruction MSE/FVU, all five objectives, and the weighted total.
- **Optimization:** actual Adam/Muon update LRs, combined pre-clip gradient norm,
  per-loss raw/weighted gradient norms, and pairwise gradient cosines. Per-loss
  diagnostics use a fixed training batch before each epoch's updates.
- **Progress:** documents, tokens, prefix-length histogram, timings, and ETA.
  Console heartbeats default to 60 seconds; W&B heartbeats update summary only,
  never adding quality-history steps. Detailed metrics are recorded without
  automatically expanding every diagnostic into a dashboard panel.

FVU divides accumulated squared error by epoch-wide centered target variation;
it is **not an average of per-layer or per-example ratios**. `train/update_*`
metrics separately describe averages during optimizer updates.

Each completed epoch writes `resume.pt`. `best.pt` minimizes the configured
weighted **`val/loss`**; `final.pt` is the actual final epoch. Resume with the
same recipe:

```bash
uv run --locked python -m kae.train --config configs/gemma3-4b.toml --resume
```

Resume restores model/normalization, both optimizers and schedulers, RNG, source
cursor/shuffle, pending targets, fixed probe, progress, and W&B identity/history.
The epoch target is total, not additional. Send `SIGUSR1` or create
`output_dir/pause.request` to finish the epoch, validate, save, and exit with code
75. A hard kill may require replaying the uncommitted epoch.

`initial_checkpoint` is a compatible weights-only warm start into a new run.
Existing version-2 student weight checkpoints remain supported. Full resume
files from the removed experimental pipeline are intentionally unsupported;
clean streaming uses resume/stream schema version 2. Existing checkpoints,
datasets, and research results outside `kae/` are retained.

## Original A-LQR / S-PID baseline

`steer/` remains the upstream collection/control/concept/toxicity/truthfulness/
refusal code for later integration. It is not part of the streaming trainer.

```bash
uv sync --locked --group baseline
uv run --locked --group baseline python -c 'from steer.data_handling import ContrastiveBuilder'
```

For the Gemma-2-2B toxicity reproduction, use the two thin launchers:

```bash
bash scripts/paper_alqr_toxicity.sh
bash scripts/paper_spid_toxicity.sh
```

They call the released functions in `steer/`, using all layers, last-token
feedback, and the original absolute LFS setpoint. No KAE, standardization, or
replacement controller is involved. Defaults are five trials of 1,000 random
RTP prompts across toxicity levels, classifier toxicity and Dist-1–3, 4-bit
Mistral-7B perplexity, and five-shot MMLU on 1,000 sampled questions per setting.

- A-LQR uses the existing `test_toxicity.py` main: `q=0.1`, `r=10`, `qf=1`,
  `lambda=3.5`.
- S-PID calls the existing `run_trials` with the Gemma-2-2B settings released in
  `tox_evalPID.py`: `kp=0.7`, `ki=0.01`, `kd=0.1`, `lambda=[0.5, 1]`, at
  five-by-1,000 scale rather than that wrapper's ten-prompt demonstration.

Both reuse `gemma-2-2b-tox.pkl` and `gemma-2-2b-nontox.pkl`; A-LQR also needs
`gemma-2-2b-nontox_jac.pkl`. Their location is `pickle_jar` in
`steer/config/config.yaml` (by default `~/pickles/` in this checkout). If absent,
collect them using the original command:

```bash
uv run --locked --group baseline python -m steer.toxicity.tox_data_script --model gemma2b
```

Results are written under `steer/toxicity/tox_data/` (ignored by Git). The
A-LQR generation file keeps the upstream misnomer `gemma-2-9b-tox-prelim.txt`,
despite loading Gemma-2-2B; its scored copy ends in `_withPPL.txt`. S-PID writes
`gemma-2-2b-spid-table1.txt` and its `_withPPL.txt` copy. MMLU results use
`gemma-2-2b-{alqr,spid}-mmlu.json`. Run the launchers sequentially on one GPU.
The printed Dist scores come from upstream; the A-LQR JSON does not save them.

These reproduce the released methods and settings, not a guarantee of matching
Table 1's selected numerical scores. The paper leaves the acceptable PPL margin
unspecified; the scripts do not invent a selection threshold. Sampling remains
as implemented upstream, including possible overlap across trials.

**Toxicity notebook (Gemma-2-2B):**
[notebooks/paper_toxicity.ipynb](notebooks/paper_toxicity.ipynb) runs the same
original baselines interactively:

```bash
uv run --frozen --group baseline --with jupyterlab --with ipykernel jupyter lab notebooks/paper_toxicity.ipynb
```

Its default is 10 RTP prompts, one repeat, and the released 100-token generation
protocol. It reports classifier toxicity, transition counts, and Dist-1–3;
original Mistral-7B perplexity is enabled, while optional five-shot MMLU is off.
Edit `N_PROMPTS` / `N_REPEATS` to increase the run size. Calibration is reused
from the toxicity pickles described above. Results are saved under
`steer/toxicity/tox_data/notebook-n10-r1/` with the default settings.

Both notebooks use bordered score tables and separate prompt/response panels,
with each controller and lambda labeled. Raw upstream logs are expandable rather
than mixed into the answer comparisons. Setup and reporting code is collapsed
by default; the run-size controls and custom-prompt cell remain visible.

**TruthfulQA notebook (Gemma-2-2B):**
[notebooks/paper_truthfulness.ipynb](notebooks/paper_truthfulness.ipynb) walks
through unsteered generation, the released A-LQR and S-PID functions, and the
original truthfulness/informativeness judges. From the repository root:

```bash
# Once, if the three truthfulness calibration pickles do not already exist:
bash scripts/paper_truthfulness_collect.sh

uv run --frozen --group baseline --with jupyterlab --with ipykernel jupyter lab notebooks/paper_truthfulness.ipynb
```

The collector writes `gemma-2-2b-{true,false,true_jac}.pkl` under `pickle_jar`,
plus a JSON record of its seed and sampled prompts. It refuses to overwrite
existing calibration files. The recipe uses 200 truthful and 200 false QA pairs,
50 truthful-prompt Jacobians, and a 24-token Jacobian context limit. These are
explicit reproduction settings; the paper does not publish exact calibration
counts.

Run cells top to bottom with **Shift+Enter**, or use **Kernel → Restart Kernel
and Run All Cells**. The default is a 10-question demonstration with one repeat;
increase `N_QUESTIONS` and `REPEATS` in the notebook for larger runs. Answers,
judge scores, and a summary CSV are saved under
`steer/truthfulness/tqa_data/` (ignored by Git). The notebook explains the
controller settings, score interpretation, a custom-question example, and
optional MMLU evaluation. Shut down its kernel afterward to release GPU memory.
`--frozen` uses the existing lockfile without rewriting training dependencies.

Both controllers accept `track_setpoint(..., layer_inds=...)`: `None` selects
all blocks (the default), `range(T - 4, T)` selects the final four of `T` blocks,
and `[]` disables interventions. Keep full-length `E` and `A`; indices are never
renumbered. Inactive blocks remain observed so S-PID retains its existing
history/reset schedule, and `U` is cleared between runs. A-LQR reuses its
full-horizon gains: restricting blocks is a location ablation, not an
energy-matched comparison or a reoptimized restricted-actuator controller.

Set paths in `steer/config/config.yaml` before those experiments. See
[the upstream repository](https://github.com/trustworthyrobotics/lqr-activation-steering)
and [the A-LQR paper](https://arxiv.org/abs/2604.19018). Retention does not establish
that every upstream wrapper supports our Gemma3 suffix.

