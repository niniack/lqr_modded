#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

uv run --locked --offline python - <<'PY'
import hashlib
import json
import math
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Subset

from kae.models import (
    AIKAE,
    KoopmanOperator,
    LayerwiseKoopmanOperator,
    SharedModesKoopmanOperator,
)
from kae.schedules import cosine_warmup_scheduler
from kae.teacher import load_teacher_head
from kae.train import load_training_data, train_one_epoch, validate_one_epoch

cache_dir = Path("artifacts/cache/autoresearch-gemma3-270m")
manifest_bytes = (cache_dir / "manifest.json").read_bytes()
manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
if manifest_sha != "f5780921d8602c7fc362b50062af52652075ab5ab1c941b12a40f1015c90b902":
    raise ValueError("Benchmark requires the verified, frozen Gemma 3 270M cache")
manifest = json.loads(manifest_bytes)
if (
    manifest["num_transformer_layers"] != 18
    or manifest["hidden_width"] != 640
    or manifest["cached_layers"] != [14, 15, 16, 17, 18]
    or manifest["splits"]["train"]["documents"] != 16384
    or manifest["splits"]["val"]["documents"] != 512
    or manifest["splits"]["test"]["documents"] != 512
    or manifest["jvp"]["directions_per_document"] != 4
    or manifest["jvp"]["final_model_norm_applied"]
    or manifest["seed"] != 1729
):
    raise ValueError("Gemma 3 270M teacher, endpoint, split, or direction mismatch")
if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires CUDA")

config = json.loads(Path("autoresearch_config.json").read_text())
loss_weights = {key: value for key, value in config.items() if key.endswith("_weight")}
if not loss_weights or any(not math.isfinite(v) or v < 0 for v in loss_weights.values()):
    raise ValueError("Loss weights must be finite and nonnegative")
if sum(loss_weights.values()) == 0:
    raise ValueError("At least one training loss must be active")
operator_classes = {
    "shared": KoopmanOperator,
    "layerwise": LayerwiseKoopmanOperator,
    "shared-modes": SharedModesKoopmanOperator,
}
teacher_name = "google/gemma-3-270m"
teacher_revision = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
output_layers = [15, 16, 17, 18]
epochs = 5
batch_size = 32

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1729)
train_data = load_training_data(cache_dir, manifest, include_jvp=True)
val_data = load_training_data(cache_dir, manifest, split="val", include_jvp=True)
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True,
    generator=torch.Generator().manual_seed(1729),
)
val_loader = DataLoader(val_data, batch_size=batch_size)
train_eval_loader = DataLoader(Subset(train_data, range(256)), batch_size=batch_size)
with torch.random.fork_rng(devices=[]):
    operator = operator_classes[config["operator"]](config["latent_width"], steps=4)
model_config = {
    "hidden_width": 640,
    "latent_width": config["latent_width"],
    "flow_layers": config["flow_layers"],
    "flow_hidden_width": config["flow_hidden_width"],
    "mlp_width": config["mlp_width"],
}
model = AIKAE(**model_config, operator=operator).cuda()
teacher = load_teacher_head(teacher_name, teacher_revision, "cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_steps = epochs * len(train_loader)
warmup_steps = math.ceil(total_steps * 0.05)
scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)

with wandb.init(
    project="tokae",
    group="gemma3-270m-last4-autoresearch",
    name=(
        f"gemma3-270m-last4-16k-5e-{config['operator']}"
        f"-latent{config['latent_width']}-flow{config['flow_layers']}x{config['flow_hidden_width']}"
    ),
    job_type="autoresearch",
    tags=["teacher:gemma3-270m", "student:AIKAE", f"operator:{config['operator']}", "last4", "pilot5e"],
    config={
        **config,
        "teacher_model": teacher_name,
        "teacher_model_revision": teacher_revision,
        "teacher_checkpoint_path": manifest["model"],
        "student_model": "AIKAE",
        "koopman_operator": config["operator"],
        "hidden_width": 640,
        "cached_layers": manifest["cached_layers"],
        "input_layer": 14,
        "output_layers": output_layers,
        "num_transformer_layers": 18,
        "num_train_docs": len(train_data),
        "num_val_docs": len(val_data),
        "num_train_eval_docs": 256,
        "jvp_directions_used": 4,
        "batch_size": batch_size,
        "seed": 1729,
        "epochs": epochs,
        "validate_every": 1,
        "learning_rate": 1e-4,
        "lr_schedule": "linear_warmup_cosine_decay",
        "warmup_fraction": 0.05,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "cache_manifest_sha256": manifest_sha,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    },
    settings=wandb.Settings(init_timeout=60),
) as run:
    print(f"ASI wandb_run_url={run.url}", flush=True)
    print("ASI model_config=" + json.dumps(config, separators=(",", ":")), flush=True)
    artifact_dir = Path("artifacts/cache/gemma3-270m-autoresearch-runs") / run.id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(epochs + 1):
        metrics = {}
        if epoch:
            metrics.update(train_one_epoch(
                model, train_loader, optimizer, device="cuda", output_layers=output_layers,
                teacher=teacher, record_diagnostics=True, scheduler=scheduler, **loss_weights,
            ))
        validation = validate_one_epoch(
            model, val_loader, device="cuda", output_layers=output_layers,
            teacher=teacher, **loss_weights,
        )
        train_eval = validate_one_epoch(
            model, train_eval_loader, device="cuda", output_layers=output_layers,
            teacher=teacher, split="train_eval", **loss_weights,
        )
        metrics.update(validation)
        metrics.update(train_eval)
        for name in ("jvp_relative_error", "state_rollout_fvu", "latent_fvu", "kl_divergence"):
            metrics[f"gap/{name}"] = validation[f"val/{name}"] - train_eval[f"train_eval/{name}"]
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite training or evaluation metric at epoch {epoch}")
        row = {"epoch": epoch, **metrics}
        history.append(row)
        run.log(row, step=epoch)
        print("EPOCH " + json.dumps({
            "epoch": epoch,
            **{key: value for key, value in metrics.items() if key.startswith(("val/", "gap/"))},
        }, separators=(",", ":")), flush=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {**model_config, "operator": config["operator"], "operator_steps": 4},
        "teacher_model": teacher_name,
        "teacher_model_revision": teacher_revision,
        "cached_layers": manifest["cached_layers"],
        "training_config": config,
        "epoch": epochs,
        "metrics": history[-1],
    }, artifact_dir / "student.pt")
    run.summary["benchmark_epoch"] = epochs
    run.summary["benchmark_checkpoint"] = str(artifact_dir / "student.pt")
    print(f"ASI results_path={artifact_dir / 'metrics.json'}", flush=True)
    for name, value in (
        ("val_jvp_relative_error", validation["val/jvp_relative_error"]),
        ("val_state_fvu", validation["val/state_rollout_fvu"]),
        ("val_latent_fvu", validation["val/latent_fvu"]),
        ("val_kl_divergence", validation["val/kl_divergence"]),
        ("jvp_generalization_gap", metrics["gap/jvp_relative_error"]),
        ("state_generalization_gap", metrics["gap/state_rollout_fvu"]),
        ("latent_generalization_gap", metrics["gap/latent_fvu"]),
        ("kl_generalization_gap", metrics["gap/kl_divergence"]),
    ):
        print(f"METRIC {name}={value:.9g}")
PY
