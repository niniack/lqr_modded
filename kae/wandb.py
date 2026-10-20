"""Shared provenance and epoch-based W&B reporting for training entrypoints."""

import hashlib
import json
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run


LOGGING_SCHEMA = "train-val-epoch-v2"
TRAIN_METRIC_DEFINITION = (
    "Post-update evaluation of the fixed first min(256, num_train_docs) training "
    "documents at the same checkpoint as validation; epoch 0 is initialization."
)
CORE_METRICS = (
    "jvp_relative_error",
    "state_rollout_fvu",
    "latent_fvu",
    "kl_divergence",
    "reconstruction_fvu",
    "loss",
)


def cache_run_config(cache_dir: Path, manifest: dict) -> dict:
    """Describe the immutable cache, optionally enriched by verified provenance."""
    manifest_bytes = (cache_dir / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    provenance_path = cache_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if (
        provenance.get("cache_manifest_sha256") is not None
        and provenance["cache_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError(f"Provenance does not match cache manifest: {provenance_path}")
    layers = manifest["cached_layers"]
    jvp = manifest.get("jvp")
    directions = 0 if jvp is None else jvp["directions_per_document"]
    config = {
        "cache_dir": str(cache_dir.resolve()),
        "cache_manifest_sha256": manifest_sha256,
        "cache_manifest": manifest,
        "cache_provenance": provenance,
        "cache_dataset": manifest.get("dataset"),
        "cache_dataset_config": manifest.get("dataset_config"),
        "dataset": provenance.get("dataset", manifest.get("dataset")),
        "dataset_config": provenance.get("dataset_config", manifest.get("dataset_config")),
        "dataset_split": provenance.get("dataset_split", manifest.get("dataset_split")),
        "text_column": manifest.get("text_column"),
        "max_length": manifest.get("max_length"),
        "data_files": manifest.get("data_files") or provenance.get("source_data_file"),
        "source_data_sha256": provenance.get("source_data_sha256"),
        "teacher_model": provenance.get("teacher_model", manifest["model"]),
        "teacher_model_revision": provenance.get(
            "teacher_model_revision",
            provenance.get("revision", manifest.get("model_revision")),
        ),
        # Identity may come from provenance; loading must retain the cached weights.
        "teacher_checkpoint_path": manifest["model"],
        "teacher_checkpoint_revision": manifest.get("model_revision"),
        "cached_layers": layers,
        "input_layer": layers[0],
        "output_layers": layers[1:],
        "num_transformer_layers": manifest["num_transformer_layers"],
        "hidden_width": manifest["hidden_width"],
        "cached_jvp_directions": directions,
        "jvp_directions_used": directions,
        "jvp_directions_optimized": directions,
    }
    config.update(
        {f"num_{split}_docs": info["documents"] for split, info in manifest["splits"].items()}
    )
    return config


def init_run(config: dict, *, name: str | None = None, group: str | None = None) -> Run:
    """Start a new-schema run with only the twelve core split curves visible."""
    run = wandb.init(
        project="tokae",
        name=name,
        group=group,
        config={
            **config,
            "logging_schema": LOGGING_SCHEMA,
            "train_metric_definition": TRAIN_METRIC_DEFINITION,
            "num_train_eval_docs": min(256, config.get("num_train_docs", 0)),
        },
    )
    try:
        run.define_metric("epoch", hidden=True)
        run.define_metric("*", step_metric="epoch", hidden=True)
        for split in ("train", "val"):
            for metric in CORE_METRICS:
                run.define_metric(f"{split}/{metric}", step_metric="epoch", hidden=False)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def log_epoch(run: Run, epoch: int, metrics: dict) -> None:
    """Commit one checkpoint row at its exact SDK step; retain other data in summary."""
    history = {"epoch": epoch}
    visible = {f"{split}/{metric}" for split in ("train", "val") for metric in CORE_METRICS}
    for key, value in metrics.items():
        if key == "epoch":
            continue
        if key.startswith(("train/", "val/")) or key == "optimizer_step":
            run.define_metric(key, step_metric="epoch", hidden=key not in visible)
            history[key] = value
        else:
            run.summary[key] = value
    for metric in CORE_METRICS:
        if f"train/{metric}" in history and f"val/{metric}" in history:
            run.summary[f"generalization_gap/{metric}"] = (
                history[f"val/{metric}"] - history[f"train/{metric}"]
            )
    run.log(history, step=epoch, commit=True)
