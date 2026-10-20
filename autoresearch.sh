#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Copy the validated cache once to this ignored path; benchmarking never contacts HPC.
cache_dir="${AUTORESEARCH_CACHE_DIR:-artifacts/cache/autoresearch-gemma2}"
test -r "$cache_dir/manifest.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

uv run --locked --offline python - "$cache_dir" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kae.models import AIKAE, KoopmanOperator
from kae.schedules import cosine_warmup_scheduler
from kae.teacher import load_teacher_head
from kae.train import load_training_data, train_one_epoch, validate_one_epoch

cache_dir = Path(sys.argv[1])
manifest = json.loads((cache_dir / "manifest.json").read_text())
if (
    manifest["cached_layers"] != [14, 15, 16, 17, 18]
    or manifest["splits"]["train"]["documents"] != 4096
    or manifest["splits"]["val"]["documents"] != 128
    or manifest["jvp"]["directions_per_document"] != 4
):
    raise ValueError("Benchmark requires the validated Gemma2 FineWeb cache")
if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires CUDA")

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1729)
train_data = load_training_data(cache_dir, manifest, include_jvp=True)
val_data = load_training_data(cache_dir, manifest, split="val", include_jvp=True)
train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True,
    generator=torch.Generator().manual_seed(1729),
)
val_loader = DataLoader(val_data, batch_size=32)
model = AIKAE(2304, 2304, operator=KoopmanOperator(2304, steps=4)).cuda()
teacher = load_teacher_head(
    "google/gemma-2-2b", "c5ebcd40d208330abc697524c919956e692655cf", "cuda"
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 5
total_steps = epochs * len(train_loader)
scheduler = cosine_warmup_scheduler(optimizer, total_steps, math.ceil(total_steps * 0.05))
for _ in range(epochs):
    training = train_one_epoch(
        model, train_loader, optimizer, device="cuda", output_layers=[15, 16, 17, 18],
        state_weight=1, jvp_weight=10000, kl_weight=1, teacher=teacher,
        record_diagnostics=True, scheduler=scheduler,
    )
validation = validate_one_epoch(
    model, val_loader, device="cuda", output_layers=[15, 16, 17, 18],
    teacher=teacher, state_weight=1, jvp_weight=10000, kl_weight=1,
)
for name, value in (
    ("val_jvp_relative_error", validation["val/jvp_relative_error"]),
    ("val_state_fvu", validation["val/state_rollout_fvu"]),
    ("val_latent_fvu", validation["val/latent_fvu"]),
    ("val_kl_divergence", validation["val/kl_divergence"]),
    ("jvp_generalization_gap", validation["val/jvp_relative_error"] - training["train/jvp_relative_error"]),
):
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {value}")
    print(f"METRIC {name}={value:.9g}")
PY
