# A-LQR with measured drift cancellation

**Start with [notebook 03](03_drift_lqr_ablation.ipynb).** It opens the completed Qwen3-4B/football experiment without loading models or running generation. Both retained notebooks default to `RUN = False` and include saved outputs.

| Notebook | Purpose |
|---|---|
| [02](02_qwen3_model_ablation.ipynb) | Identity / Mean Jacobian / Random gains across steering strength and actuation interval. |
| [03](03_drift_lqr_ablation.ipynb) | Measured cancellation off/on, all / last-8 / last-4 blocks, and Steering DMD gains. |

The large run contains **385,000 completions**: 1,000 per condition plus one shared 1,000-completion unsteered baseline. These are stochastic continuations of **one prefix, “Once upon a time”**, not 1,000 distinct prompts. Notebook 03 opens at `R=10`; change `PLOT_R` to `1` to inspect the other saved penalty. Activation shifts are within-model diagnostics, not interchangeable semantic strength across models.

## uv setup

```bash
uv sync --locked
uv run python -m ipykernel install --user --name kae-steering --display-name "A-LQR (uv)"
uv run --with jupyterlab jupyter lab
```

Select **A-LQR (uv)** in Jupyter, or `.venv/bin/python` in your editor. `extensions` and `steer` are installed with the project; no notebook `sys.path` edits are needed. Reviewing saved tables and figures does not require a GPU or model downloads.

## Upstream and extensions

[`steer/steering.py`](steer/steering.py) and [`steer/lqr_utils.py`](steer/lqr_utils.py) are copied **unchanged** from [trustworthyrobotics/lqr-activation-steering at `c2e0c845`](https://github.com/trustworthyrobotics/lqr-activation-steering/tree/c2e0c8450797e0dc234b2c53475267a2cfb2457a). `QwenLQRSteering` subclasses the original `LQRSteering`: it executes the original `hook_setpoint_tracking`, inherits hook registration and context cleanup, and calls the original `time_varying_lqr_noB_mem_efficient`. Other upstream files are omitted.

| Extension | Responsibility |
|---|---|
| `steering.py` | Qwen compatibility for the original controller, selective actuation, measured cancellation, sampling, and replay metrics. |
| `data.py` | Qwen loading, block-input capture, and balanced contrastive concept directions. |
| `jacobian.py` | Chunked float32 last-token Jacobians for current Transformers. |
| `dmd.py` | Fit/load the retained steering-response DMD gain operator. |
| `judge.py` | The frozen Granite concept/coherence criteria and batched binary judgments. |
| `experiments.py` | Model-specific caches, resumable generation/judging, and summaries shared by 02–03. |

Measured cancellation subtracts `w K_l d_perp` from the original A-LQR push, using a separate unsteered model on the **same generated tokens**. It changes the feedback error, not the Riccati costs. `w=0` reproduces the base controller. Full gains are stored only for actuated blocks.

The bridge is necessary because current Qwen supplies block inputs by keyword and uses reduced-precision model states, while the original hook expects positional inputs with compatible dtypes. The bridge selects the actuated positions, obtains the original hook's unquantized float32 push, adds measured cancellation if enabled, and casts the push into the model output. It contains no separate A-LQR feedback formula.

The adapter carries and symmetrizes float64 cost-to-go between one-stage upstream solves, preserving the result-producing recurrence and bounded matrix workspace. Unactuated blocks still propagate their dynamics. Upstream's API discards cost-to-go, so each call does some duplicate setup computation. The constructor avoids allocating unused per-layer dense Q/R matrices. No Riccati solve occurs during generation.

Sampling retains `temperature=0.7`, `top_p=0.95`, repetition penalty `1.2`, and 96 new tokens. We do not call upstream `track_setpoint`, which hardcodes `top_p=0.3`.

The frozen implementation precomputed `K_l v_l` in float64. The original hook instead computes the full-matrix float32 product, so rounding can differ. Short seeded Qwen3-4B checks matched the frozen token IDs at `w=0` and `w=2`; this is not a claim of bitwise reproduction of the whole sweep. The saved results are unchanged, and their exact producing implementation remains archived.

## Another concept or model

Edit the single configuration cell in 02 or 03 and set `RUN = True`. For a smaller additional run in 03, change its `Sweep` configuration, for example:

```python
config = Sweep(
    notebook='03_drift_lqr_ablation', model_name='Qwen/Qwen3-1.7B',
    concept='dog', trials=20, last_n=(None,), r_values=(10.,),
    generation_batch=20, trace_batch=20, judge_batch=64,
)
```

- Concepts come from `data/filtered_sentences.csv`, including `football`, `dog`, and `radio`. The retained collection and steering code targets Qwen; other architecture adapters and the demonstration notebooks were removed.
- Directions and Jacobians are selected by model, concept, sample count, context length, and seed. Missing ones are computed once; defaults are 200 direction samples and 50 Jacobian samples.
- Random uses seed 0 and matches each Mean-Jacobian residual's Frobenius norm. Steering DMD fits fixed Identity-A-LQR push responses in float32; the saved football fit uses seed 2, 900 training / 100 validation prompts, and selected rank 128. A new DMD fit is required for a new model/concept. Omit `Steering DMD` from `operators` to skip it.
- Gains are recomputed for each operator, penalty, layer scope, and interval. Sampling resets `torch.manual_seed(seed + lambda_index)` for each condition, then advances across batches.
- Each completed condition and judge chunk is checkpointed. `generate_run(config)` and `judge_run(config)` are separate: judging can resume without loading the generation models. Both show progress bars.
- Results live under `artifacts/<model>/<notebook>/review/<concept>_n<trials>/`. Existing settings are checked before resuming; use `run_name='football_seed1_n1000'` for a deliberately different run at the same model/concept/sample count.
- Choose smaller generation/replay/judge batches before starting a larger model. Measured cancellation needs a second model copy. No automatic precision downgrade is used for DMD fitting.

## Results and caches

Small summaries, matched-concept tables, figures, and configuration are included in the review tree. Full judged tables are `results.csv.gz`; operator tensors live in `artifacts/<model>/operators/`. `index.json` records the published operator hashes and their original locations.

The local **`artifacts/review-data.tar.gz`** bundle (about 979 MiB) contains the full Qwen completion tables and required E / Mean-Jacobian / Steering-DMD caches. It is excluded from Git; attach it to the repository's release when sharing full data. Extract it from the repository root:

```bash
tar -xzf /path/to/review-data.tar.gz
```

The bundled caches are not needed to read the notebooks or regenerate their figures. Existing local runs and checkpoints were preserved outside the published review directories; none is required by the review notebooks.

## Provenance and checks

The implementation that produced the saved results is frozen at **`6c2e5b9`**, tag **`pre-review-cleanup`**, branch **`archive/pre-review-cleanup`**. Each published run's `config.json` identifies the original CSV and its SHA-256. Publication adds consistent condition columns and zero-valued unsteered deviation metrics; it does not regenerate or rejudge the expensive run.

```bash
uv run pytest -q tests/test_basics.py
```

Verified after the cutover: 19 tests; both notebooks executed offline from saved results without importing Torch or Transformers; real Qwen generation/replay, zero-weight token equality, zero-drift nominal replay, new direction/Jacobian collection, batched Granite judging, and DMD fitting/cache reuse. All 433 original artifact files retained their SHA-256 hashes. The unmodified upstream hook emits one PyTorch deprecation warning for transposing a one-dimensional vector; it does not affect these checks.

Upstream has no license file at the pinned revision. Attribution here is not a redistribution license; confirm permission with its author before publicly distributing the vendored source.
