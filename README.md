## Setup

```bash
# Install the locked dependencies into .venv (uv uses Python 3.11 from .python-version)
uv sync --locked
```

Update `steer/config/config.yaml` with the desired file paths. For concept judging, also download the NLTK WordNet corpus with `uv run python -m nltk.downloader wordnet`.

---

## KAE Training

`kae.train` loads aligned activation and optional JVP shards from both `train/` and `val/` in `--cache-dir`, infers the hidden width and layer transitions from `manifest.json`, and trains AIKAE rollouts from the first cached layer through every later cached layer. The objective weights raw state-rollout MSE, JVP MSE, and final-endpoint KL(teacher || student) separately. KL and W&B teacher-head diagnostics require the cached endpoint to be the teacher's **final transformer block**; applying its head to an intermediate layer would not produce the teacher's actual logits. Evaluation runs at initialization (epoch 0), every `--validate-every` epochs (default 1), and always at the final epoch, without optimizer updates; test shards are not used. Reconstruction and latent error are diagnostics, **not** optimized losses.

```bash
uv run python -m kae.train \
  --cache-dir artifacts/cache/gemma3-270m-fineweb \
  --latent_width 640 --epochs 10 --batch-size 32 \
  --learning_rate 1e-4 --validate-every 1 \
  --state-weight 1.0 --jvp-weight 1.0 --kl-weight 1.0 \
  --checkpoint-path artifacts/checkpoints/gemma3-270m-best.pt \
  --final-checkpoint-path artifacts/checkpoints/gemma3-270m-final.pt --wandb
```

`--checkpoint-path` retains the best evaluated validation checkpoint (including the epoch-0 baseline) by overall JVP relative error when JVP targets exist, otherwise by weighted validation loss. Each improvement atomically replaces the job's checkpoint. Optional `--final-checkpoint-path` independently saves the actual final-epoch student, never a copy of the best checkpoint; the two paths must differ, and either existing destination is refused at startup. Both files contain the student `model_state_dict`, model/operator configuration, epoch, matched-checkpoint metrics, and full run configuration including teacher/cache provenance, loadable with `torch.load(path, map_location="cpu", weights_only=True)`. The best file additionally records its selection metric and score, and W&B records its best epoch and score. The test split remains untouched for a single final evaluation after model selection.

`--wandb` opts into the `tokae` project. Cached JVPs default to weight 1 when present (otherwise 0); `--jvp-weight 0` **removes JVP from the objective but still measures it** on cached directions for both train and validation. KL defaults to weight 1; `--kl-weight 0` disables its objective contribution, but with W&B it remains a logged diagnostic. KL training or W&B logging loads the frozen FP32 **final norm and LM head** onto the training device, not the transformer blocks. The loader transiently loads the full checkpoint on CPU, so it still needs CPU memory for the checkpoint and device memory for the vocabulary-sized head and `[batch, vocabulary]` logits. `--seed 1729` (default) fixes student initialization and the independent training-shuffle generator for comparable ablations. At initialization the W&B config records that seed, batch size, teacher and student model names, train- and validation-document counts, cached/measured/optimized JVP-direction counts, cached layers, learning rate, and all three loss weights.

All training entrypoints use `kae.wandb` for cache provenance, W&B initialization, and epoch logging. New runs explicitly use `logging_schema=train-val-epoch-v2`: only `train/*` and `val/*` are history metric namespaces, and the six visible curves per split are `jvp_relative_error`, `state_rollout_fvu`, `latent_fvu`, `kl_divergence`, `reconstruction_fvu`, and `loss` (where targets exist). Raw MSEs, per-layer details, and per-epoch `train/grad_norm/*` remain available as hidden history metrics; `epoch`, `optimizer_step`, and `train/learning_rate` are hidden scalars. Every history row uses both `epoch=N` and SDK `step=N`, starting at zero; all charts use the epoch axis. Config retains `epochs`, batch size, seed, loss weights, architecture/parameter counts, and exact scheduler settings. Existing historical runs are not relabeled or rewritten: their old pre-update training-batch rows must not be interpreted as the new checkpoint evaluations.

For fixed-update-budget studies, `epochs` and `epoch` count actual data passes: optimizer updates divided by batches per pass. For example, 640 updates at batch size 32 mean 5 epochs on 4,096 documents but 1.25 epochs on 16,384 documents. All-layer reconstruction, when optimized in an auxiliary ordinary-KAE comparison, is reported separately as `reconstruction_all_layers_mse` with its own `loss_weights.reconstruction_all_layers_mse`; `reconstruction_mse` / `reconstruction_fvu` continue to mean input-state reconstruction.

Under the new schema, `train/*` evaluates the fixed first `min(256, num_train_docs)` cached training documents under `torch.no_grad()`, at the **same frozen checkpoint** as `val/*`. Epoch 0 measures initialization without an optimizer update; epochs 1 through `--epochs` each perform exactly one training pass. Evaluations occur at epoch 0, each `--validate-every` boundary, and the final epoch; intermediate rows contain only hidden optimizer-step/LR scalars and available gradient norms, not fabricated split measurements. Config explicitly records `train_metric_definition` and `num_train_eval_docs`. The fixed subset and held-out validation split provide diagnostics, not a guarantee of generalization. There are no new-schema `train_eval/*` or `gap/*` history namespaces; matched-checkpoint validation-minus-training gaps are retained in W&B summary under `generalization_gap/`.

`--learning_rate` sets the peak rate (default `1e-4`); the optimizer warms up linearly for the first 5% of **total optimizer steps**, then follows a cosine decay to zero. Config records total steps and the exact warmup step count. Hidden `train/learning_rate` is the rate used on the last optimizer update (or the initial scheduled rate at epoch 0). Under both splits, `state_mse_first` measures the first predicted layer, `state_mse/layer_<N>` each predicted layer, and `state_rollout_mse` all predicted layers. `state_fvu_first`, `state_fvu/layer_<N>`, and `state_rollout_fvu` divide total squared error by the split-wide target variation about each layer/feature's mean. `latent_mse` and `latent_fvu` compare evolved student latents to the student's encoder applied to cached future teacher states; these moving targets are diagnostics only. `kl_divergence` compares final-endpoint logits as KL(teacher || student).

`train/reconstruction_mse` and `val/reconstruction_mse` compare `decode(encode(x))` with the cached input activation `x`; `reconstruction_fvu` divides total squared error by each split's input variation around its per-feature mean. AIKAE's invertible flow should reconstruct almost exactly without a reconstruction loss.

When JVP targets are cached, both splits retain raw `jvp_mse`, `jvp_mse/layer_<N>`, overall `jvp_relative_error`, and `jvp_relative_error/layer_<N>` even with `--jvp-weight 0`. Relative error remains the ratio of **sums** of squared JVP errors and squared teacher responses across examples and directions (and across layers for the overall metric), with `1e-8` added to the denominator; response vectors are not normalized. With `--wandb`, pre-update training-batch losses and gradient diagnostics are retained in the `training_batch_metrics` summary dictionary, never merged into checkpoint `train/*` evaluation curves; only `train/grad_norm/*` additionally retains hidden per-epoch history. Its `train/grad_norm/{total,flow,operator,augmentation}` values (augmentation when present) are mean L2 norms of the weighted combined objective after backward and before updates. `train/grad_norm_by_loss/{state_mse,jvp_mse,kl_divergence}/{total,flow,operator,augmentation}` records separate **unweighted** gradients on the first pre-update batch, including zero-weight components when targets exist; the zero-weight JVP arm otherwise measures responses without gradients. Validation has no gradient metrics. Without `--wandb`, checkpoint evaluations and console losses still run, but no metrics are sent to W&B.

`cache_run_config` hashes the frozen manifest and records its path, dataset/config/split, text column, truncation length, cached/input/output layers, transformer depth, document counts, and cached/measured/optimized JVP directions. New cache manifests also retain `data_files`; frozen manifests without that field use the provenance sidecar's `source_data_file` when available. An optional adjacent `provenance.json` supplies verified original dataset/config, source-data SHA-256, and teacher identity/revision when a manifest describes local JSON or local model snapshots; if it records `cache_manifest_sha256`, that hash must match the actual manifest. The untouched manifest and sidecar remain in config; the teacher checkpoint path/revision from the manifest, not a guessed public identity, is used to load frozen weights. No dataset or teacher identity is hardcoded.

`kae/losses.py` exports `mse_loss(prediction, target)` for state, latent, JVP, or reconstruction vectors. It checks matching shapes and averages over every element; cached JVP responses have shape `[directions, batch, output_layers, hidden_width]`. Student and teacher JVPs must use the same directions and layer order.

`kl_divergence_loss(student_logits, teacher_logits)` computes mean KL(teacher || student), summing over the vocabulary for each final-token distribution. The loss accepts logits, not hidden states. For Gemma 2 and Gemma 3, `kae.teacher.load_teacher_head` retains the checkpoint's frozen final norm, LM head, and optional `final_logit_softcapping` transform. It applies them to both the KAE prediction and the cached raw final-block teacher state. Teacher logits are detached; gradients pass through the frozen head to the KAE when KL training is enabled. W&B-only KL diagnostics run under `torch.no_grad()`.

Both models accept an operator module and default to a one-step `KoopmanOperator`. Training's `--operator shared` (default) repeatedly applies one linear transition; `--operator layerwise` uses independent identity-initialized linear transitions for the predicted layers; `--operator shared-modes` uses per-step growth/rotation blocks in a shared learned modal basis. All three obey the same rollout and JVP interface, and the selected operator and flow dimensions are recorded in W&B config. At a fixed seed, operator initialization preserves the random state so architecture comparisons start with the same flow and augmentation weights. `LayerwiseKoopmanOperator` and `SharedModesKoopmanOperator` can also be passed directly to `AIKAE`.

Setting `--latent_width` above the teacher's hidden width adds learned nonlinear observables to the latent state before applying the Koopman operator. At equal widths the flow is invertible without augmentation. Latent FVU across different widths measures different learned spaces and is not directly comparable; state FVU and teacher-response JVP error still refer to the same cached targets.

### Controlled Gemma 3 270M benchmark

`bash autoresearch.sh` runs the pinned Gemma 3 270M H14–H18 experiment against the verified cache at `artifacts/cache/autoresearch-gemma3-270m`. `autoresearch_config.json` selects the Koopman operator, latent/flow architecture, and separately weighted raw losses. The harness fixes the teacher revision, cache manifest, 16,384 training and 512 validation documents, four JVP directions, seed 1729, batch size 32, and ten-epoch cosine/warmup schedule.

The primary score is **final-epoch** validation JVP relative error, not a cherry-picked best epoch. Each epoch also evaluates the same first 256 training documents under `train/*`; the 512 test documents are not evaluated. Online W&B runs in `tokae` use the same central reporting schema and provenance as the CLI, and log epoch/SDK step 0 through 10. Final student checkpoints and complete metric curves are saved under `artifacts/cache/gemma3-270m-autoresearch-runs/<wandb_run_id>/`. The training CLI's optional best-validation checkpoint has a different selection rule.

## Activation Cache

`kae.cache_activations` streams a dataset and stores final-token activations and optional JVPs. Train, validation, and test documents come from one seeded, shuffled source stream without overlap.

```bash
uv run python -m kae.cache_activations \
  --model google/gemma-3-270m \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --train-docs 16384 --val-docs 512 --test-docs 512 \
  --last-layers 4 --jvp-directions 4 \
  --output-dir artifacts/cache/gemma3-270m-fineweb
```

`--last-layers 4` selects the last four transformer **blocks**, plus their input activation. Gemma 3 270M has 18 blocks: cache H14–H18 and run blocks 15–18. Gemma 3 1B has 26 blocks: cache H22–H26 and run blocks 23–26. `--layers START END` instead selects an explicit inclusive, contiguous range of 1-based outputs. JVP primals are the full sequence at the input layer; the slice returns final-token responses after each subsequent block. Directions are unit-norm Gaussian, with nonzero tangent only at each example's final valid token. JVP mode reuses the teacher forward's exact block arguments, masks, and positional embeddings (including Gemma 3's global/local RoPE), using eager attention for forward-mode AD.

Cached activations are raw block outputs before the model-wide final norm. The causal-LM model, including its output head, stays loaded, but extraction calls only `model.model`; it does not compute logits. Transformers is pinned to `4.57.1` in `pyproject.toml`. Teacher compute and floating-point cache tensors use FP32.

Each activation shard under `train/`, `val/`, or `test/` is a dictionary with `activations` shaped `[documents, cached_layers, hidden_width]`, concatenated unpadded `input_ids`, and `input_offsets` delimiting each example. JVP shards under `jvp/<split>/` contain `directions` shaped `[directions, documents, hidden_width]` and `responses` shaped `[directions, documents, output_layers, hidden_width]`. Shards hold up to 256 documents; the manifest records actual tensor dtypes and model settings. `--batch-size` controls inference batching, not shard size.

**Remote Parquet caveat:** An early-stopped FineWeb stream can trigger a PyArrow `PyGILState_Release` abort at interpreter shutdown, even after all cache shards and the manifest are written. A clean local-data run uses `--dataset json --data-files /path/to/fineweb.jsonl` with documents saved as `{"text": "..."}` JSON lines; the four-document Gemma2 pilot completed this way. Do not treat an exit-code-134 remote run as successful.

## Jacobian Collection

Each submodule includes a `<name>_data_script.py` file that handles contrastive vector construction and Jacobian collection.

These computations are best performed offline, as memory requirements can be high. If needed, you can replace calls to `collect_jacobians` with `collect_jacobians_vram` for reduced memory usage.

```bash
# Toxicity
uv run python -m steer.toxicity.tox_data_script --model=gemma2b  # replace with your model of choice

# Truthfulness
uv run python -m steer.truthfulness.tqa_data_script --model=gemma2b

# Concept Steering (see script for available concepts)
uv run python -m steer.concepts.con_data_script --model=gemma2b

# Refusal
uv run python -m steer.refusal.ref_data_script --model=gemma2b
```

---

## Evaluation

Toxicity and Truthfulness (TQA) include dedicated evaluation scripts with auxiliary benchmarks.  
Concept Steering and Refusal use slightly different evaluation pipelines.

```bash
# Toxicity
uv run python -m steer.tox_eval        # A-LQR
uv run python -m steer.tox_evalPID     # S-PID

# Truthfulness
uv run python -m steer.tqa_eval        # A-LQR
uv run python -m steer.tqa_evalPID     # S-PID

# Concept Steering (see script for available concepts)
uv run python -m steer.concepts.concept_pipeline   # A-LQR

# Score an existing concept evaluation CSV in place
uv run python -m steer.concepts.concept_score path/to/evaluation.csv dog 1.5
# Refusal
uv run python -m steer.refusal.ref_asr_script --model=gemma2b --steering=lqr  # A-LQR+
uv run python -m steer.refusal.ref_asr_script --model=gemma2b --steering=pid  # PID
```

Each submodule also includes a `test_<name>.py` script for debugging and experimentation.  
Notably, `test_ref.py` provides an out-of-the-box interactive demo.

```bash
uv run python -m steer.refusal.test_ref # A-LQR+

```