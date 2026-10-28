"""Shared provenance and epoch-based W&B reporting for training entrypoints."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


LOGGING_SCHEMA = "train-val-clean-mse-epoch-v6"
TRAIN_METRIC_DEFINITION = (
    "Post-update evaluation of the fixed first min(256, num_train_docs) training "
    "documents at the same checkpoint as validation; epoch 0 is initialization."
)
CORE_METRICS = (
    "state_rollout_fvu",
    "latent_fvu",
    "kl_divergence",
    "reconstruction_fvu",
    "loss",
)


def run_metadata(recipe, metadata, model, model_config, train_loader, reporter,
                 total_steps, warmup_steps, *, loss_weights):
    """Record the same configuration locally and on W&B, even when W&B is disabled."""
    output = Path(recipe.output_dir)
    return {
        **metadata,
        "logging_schema": LOGGING_SCHEMA,
        "train_metric_definition": TRAIN_METRIC_DEFINITION,
        "student_model": type(model).__name__,
        "architecture": model_config.get("architecture", "aikae"),
        "model_config": model_config,
        "koopman_operator": recipe.operator,
        "hidden_width": model_config["hidden_width"],
        "latent_width": model_config["latent_width"],
        "operator_steps": model_config["operator_steps"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_train_docs": len(train_loader.dataset),
        "num_train_eval_docs": len(train_loader.dataset.probe_dataset),
        "optimizer": f"Muon matrix weights x{recipe.muon_lr_multiplier} + Adam biases",
        "optimizer_recipe": {
            "weights": {"optimizer": "Muon", "lr_multiplier": recipe.muon_lr_multiplier,
                        "momentum": 0.95, "nesterov": True, "ns_steps": 5,
                        "adjust_lr_fn": "match_rms_adamw"},
            "biases": {"optimizer": "Adam", "lr_multiplier": 1.0,
                       "betas": [0.9, 0.999], "eps": 1e-8},
            "weight_decay": 0.0,
        },
        "lr_schedule_definition": "linear_warmup_cosine_decay",
        "learning_rate_definition": "Adam base LR; actual Adam and Muon update rates are logged separately",
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "loss_weights": loss_weights,
        "progress_every_seconds": reporter.interval_seconds,
        "checkpoint_path": str(output / "best.pt"),
        "final_checkpoint_path": str(output / "final.pt"),
        "checkpoint_selection": {"metric": "val/loss", "mode": "min"},
        "normalization": {
            "mode": recipe.student_normalization,
            **model_config["normalization_fit"],
            "external_coordinates": "raw teacher residual states before final normalization",
            "reconstruction_objective": "plain raw input-state MSE between decode(encode(x0)) and x0; no operator steps",
            "state_objective": "plain elementwise raw-state MSE across all rollout outputs",
            "latent_objective": "plain MSE; propagated and teacher-encoded branches both trainable",
            "multistart_objective": "plain MSE over all later-start/end pairs equally; no propagation resets",
        },
        "metric_definitions": {
            "reconstruction_mse": "raw input-state reconstruction SSE / (documents * hidden width)",
            "state_rollout_mse": "raw rollout SSE / (documents * output boundaries * hidden width)",
            "latent_mse": "latent SSE / (documents * output boundaries * latent width)",
            "fvu": "summed SSE / (epoch-wide target variation around each feature mean + 1e-8)",
            "kl_divergence": "KL(teacher || student) at the final endpoint, through the frozen norm and head",
            "train": "fixed first training documents, evaluated after updates at the same checkpoint as val",
            "val": "fixed held-out clean cache; test documents are never evaluated during training",
        },
        "gradient_diagnostics": {
            "sampling": "fixed training-probe batch before each epoch's updates; epoch0 initialization",
            "raw_norms": "per-loss L2 norms, including zero-weight losses",
            "weighted_norms": "loss weight times raw gradient norm",
            "cosines": "unweighted pairwise gradient cosine; zero-norm pairs omitted",
            "combined_gradient": "train/grad_norm_pre_clip: mean update norm before clipping",
            "gradient_clip": recipe.grad_clip,
        },
    }


def init_run(config: dict, *, name: str | None = None, group: str | None = None,
             resume_id: str | None = None) -> "Run":
    """Start or strictly resume an online run on the exact completed-epoch axis."""
    import wandb

    run = wandb.init(
        project="tokae",
        mode="online",
        **({"id": resume_id, "resume": "must"} if resume_id is not None else {}),
        name=name or config.get("run_name"),
        entity=config.get("wandb_entity"),
        dir=config["output_dir"],
        settings=wandb.Settings(init_timeout=30, **(
            {"base_url": config["wandb_base_url"]} if config.get("wandb_base_url") else {}
        )),
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


def log_epoch(run: "Run", epoch: int, metrics: dict) -> None:
    """Commit one checkpoint row at its exact SDK step; retain other data in summary."""
    # A hard kill can occur after remote commit but before local resume.pt publication.
    # Re-execute that uncommitted local epoch without sending a duplicate history row.
    if epoch < run.step:
        return
    history = {"epoch": epoch}
    visible = {f"{split}/{metric}" for split in ("train", "val") for metric in CORE_METRICS}
    visible.update(key for key in metrics
                   if key == "train/learning_rate" or key.startswith("train/learning_rate/"))
    for key, value in metrics.items():
        if key == "epoch":
            continue
        if key.startswith(("train/", "val/")) or key == "optimizer_step":
            if key == "train/prefix_length_histogram":
                import wandb
                value = wandb.Histogram(np_histogram=(value["counts"], value["edges"]))
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
