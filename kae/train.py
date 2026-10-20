"""Train an AIKAE on cached final-token teacher activations."""

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, TensorDataset
import wandb

from kae.losses import kl_divergence_loss
from kae.metrics import EpochDiagnostics, JvpMetrics, jvp_rollout
from kae.models import (
    AIKAE,
    KoopmanOperator,
    LayerwiseKoopmanOperator,
    SharedModesKoopmanOperator,
)
from kae.schedules import cosine_warmup_scheduler
from kae.teacher import load_teacher_head


def train_one_epoch(
    model: AIKAE,
    train_loader: Iterable[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    output_layers: list[int],
    state_weight: float = 1.0,
    jvp_weight: float = 0.0,
    kl_weight: float = 0.0,
    teacher: nn.Module | None = None,
    record_diagnostics: bool = False,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    """Train on state, optional JVP, and optional final-endpoint KL losses."""
    model.train()
    weighted_total = 0.0
    kl_total = 0.0
    example_count = 0
    state_errors = torch.zeros(len(output_layers), dtype=torch.float64)
    jvp_metrics = JvpMetrics(output_layers)
    diagnostics = (
        EpochDiagnostics(model, output_layers, teacher) if record_diagnostics else None
    )
    if kl_weight and teacher is None:
        raise ValueError("KL training requires a frozen teacher")

    for batch in train_loader:
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)

        measure_loss_gradients = diagnostics is not None and diagnostics.num_batches == 0
        if jvp_weight or (measure_loss_gradients and len(batch) == 4):
            if len(batch) != 4:
                raise ValueError("JVP training needs cached directions and responses")
            directions = batch[2].to(device)
            teacher_jvps = batch[3].to(device)
            predicted_states, student_jvps = jvp_rollout(model, inputs, directions)
            jvp_mse = jvp_metrics.record(student_jvps, teacher_jvps)
        else:
            predicted_states = model.rollout(inputs)
            jvp_mse = None
            if len(batch) == 4:
                with torch.no_grad():
                    _, student_jvps = jvp_rollout(model, inputs, batch[2].to(device))
                    jvp_metrics.record(student_jvps, batch[3].to(device))

        squared_state_error = (predicted_states - targets).square()
        state_rollout_mse = squared_state_error.mean()
        kl_loss = None
        if kl_weight or (measure_loss_gradients and teacher is not None):
            student_logits = teacher(predicted_states[:, -1])
            with torch.no_grad():
                target_logits = teacher(targets[:, -1])
            kl_loss = kl_divergence_loss(student_logits, target_logits)
        if diagnostics is not None:
            diagnostics.record_batch(
                inputs, predicted_states, targets, squared_state_error, kl_loss=kl_loss
            )
        if measure_loss_gradients:
            diagnostics.record_loss_gradients(
                {
                    "state_mse": state_rollout_mse,
                    "jvp_mse": jvp_mse,
                    "kl_divergence": kl_loss,
                }
            )
        loss = state_weight * state_rollout_mse
        if jvp_weight:
            loss = loss + jvp_weight * jvp_mse
        if kl_loss is not None:
            loss = loss + kl_weight * kl_loss
        loss.backward()
        if diagnostics is not None:
            diagnostics.record_gradients()
        learning_rate_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = inputs.shape[0]
        weighted_total += loss.detach().item() * batch_size
        state_errors += squared_state_error.detach().sum(
            dim=(0, 2), dtype=torch.float64
        ).cpu()
        if kl_loss is not None:
            kl_total += kl_loss.detach().item() * batch_size
        example_count += batch_size

    if example_count == 0:
        raise ValueError("train_loader must yield at least one batch")
    state_denominator = example_count * targets.shape[-1]
    metrics = {
        "train/loss": weighted_total / example_count,
        "train/learning_rate": learning_rate_used,
        "train/state_mse_first": state_errors[0].item() / state_denominator,
        "train/state_rollout_mse": state_errors.sum().item()
        / (state_denominator * len(output_layers)),
    }
    for index, layer in enumerate(output_layers):
        metrics[f"train/state_mse/layer_{layer}"] = (
            state_errors[index].item() / state_denominator
        )
    if kl_weight:
        metrics["train/kl_divergence"] = kl_total / example_count
    metrics.update(jvp_metrics.results("train", targets.shape[-1]))
    if diagnostics is not None:
        metrics.update(diagnostics.results())
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: AIKAE,
    val_loader: Iterable[tuple[torch.Tensor, ...]],
    *,
    device: torch.device | str,
    output_layers: list[int],
    teacher: nn.Module | None,
    state_weight: float,
    jvp_weight: float,
    kl_weight: float,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate cached validation targets without updating the student."""
    model.eval()
    diagnostics = EpochDiagnostics(model, output_layers, teacher)
    jvp_metrics = JvpMetrics(output_layers)
    weighted_total = 0.0
    example_count = 0
    for batch in val_loader:
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        if len(batch) == 4:
            predicted_states, student_jvps = jvp_rollout(model, inputs, batch[2].to(device))
            jvp_mse = jvp_metrics.record(student_jvps, batch[3].to(device))
        else:
            if jvp_weight:
                raise ValueError("JVP validation needs cached directions and responses")
            predicted_states = model.rollout(inputs)
            jvp_mse = None
        squared_state_error = (predicted_states - targets).square()
        kl_loss = None
        if teacher is not None:
            kl_loss = kl_divergence_loss(
                teacher(predicted_states[:, -1]), teacher(targets[:, -1])
            )
        diagnostics.record_batch(
            inputs, predicted_states, targets, squared_state_error, kl_loss=kl_loss
        )
        loss = state_weight * squared_state_error.mean()
        if jvp_weight:
            loss = loss + jvp_weight * jvp_mse
        if kl_weight:
            if kl_loss is None:
                raise ValueError("KL validation requires a frozen teacher head")
            loss = loss + kl_weight * kl_loss
        batch_size = inputs.shape[0]
        weighted_total += loss.item() * batch_size
        example_count += batch_size

    if example_count == 0:
        raise ValueError(f"{split}_loader must yield at least one batch")
    metrics = {f"{split}/loss": weighted_total / example_count}
    metrics.update(diagnostics.results(split))
    metrics.update(jvp_metrics.results(split, targets.shape[-1]))
    return metrics


def train(
    hidden_width: int,
    latent_width: int,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-4,
    device: torch.device | str | None = None,
    *,
    operator_steps: int = 1,
    operator_name: str = "shared",
    state_weight: float = 1.0,
    jvp_weight: float = 0.0,
    kl_weight: float = 0.0,
    output_layers: list[int] | None = None,
    val_loader: Iterable[tuple[torch.Tensor, ...]] | None = None,
    validate_every: int = 1,
    wandb_logging: bool = False,
    run_config: dict | None = None,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    """Train an AIKAE on weighted state, JVP, and KL losses."""
    if (
        min(state_weight, jvp_weight, kl_weight) < 0
        or state_weight + jvp_weight + kl_weight == 0
    ):
        raise ValueError("Loss weights must be nonnegative, with at least one positive")
    if validate_every < 1:
        raise ValueError("validate_every must be positive")
    if checkpoint_path is not None:
        if val_loader is None:
            raise ValueError("Checkpoint selection requires validation data")
        if checkpoint_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
    operator_classes = {
        "shared": KoopmanOperator,
        "layerwise": LayerwiseKoopmanOperator,
        "shared-modes": SharedModesKoopmanOperator,
    }
    if operator_name not in operator_classes:
        raise ValueError(f"Unknown Koopman operator: {operator_name}")

    # Operator sizes must not change the shared encoder's seeded initialization.
    with torch.random.fork_rng(devices=[]):
        operator = operator_classes[operator_name](latent_width, steps=operator_steps)
    surrogate_model = AIKAE(
        hidden_width=hidden_width,
        latent_width=latent_width,
        operator=operator,
    ).to(device)
    optimizer = Adam(surrogate_model.parameters(), lr=learning_rate)
    total_steps = epochs * len(train_loader)
    warmup_steps = math.ceil(0.05 * total_steps)
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)

    run = None
    teacher = None
    if wandb_logging or kl_weight:
        metadata = run_config or {}
        teacher_name = metadata.get("teacher_model")
        if not teacher_name:
            raise ValueError(
                "KL training or W&B diagnostics require run_config['teacher_model']"
            )
        teacher = load_teacher_head(teacher_name, metadata.get("teacher_model_revision"), device)
    if wandb_logging:
        run = wandb.init(
            project="tokae",
            config={
                **(run_config or {}),
                "student_model": type(surrogate_model).__name__,
                "koopman_operator": operator_name,
                "flow_layers": len(surrogate_model.flow),
                "flow_hidden_width": surrogate_model.flow[0].net[0].out_features,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lr_schedule": "linear_warmup_cosine_decay",
                "warmup_fraction": 0.05,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "validate_every": validate_every,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "num_train_eval_docs": min(256, len(train_loader.dataset))
                if val_loader is not None
                else 0,
                "loss_weights": {
                    "state_mse": state_weight,
                    "jvp_mse": jvp_weight,
                    "kl_divergence": kl_weight,
                },
            },
        )
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
    train_eval_loader = (
        DataLoader(
            Subset(train_loader.dataset, range(min(256, len(train_loader.dataset)))),
            batch_size=train_loader.batch_size or 32,
        )
        if wandb_logging and val_loader is not None
        else None
    )
    layers = output_layers or list(range(1, operator_steps + 1))
    best_val_metric = float("inf")

    try:
        for epoch in range(epochs):
            metrics = train_one_epoch(
                model=surrogate_model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                output_layers=layers,
                state_weight=state_weight,
                jvp_weight=jvp_weight,
                kl_weight=kl_weight,
                teacher=teacher,
                record_diagnostics=wandb_logging,
                scheduler=scheduler,
            )
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                metrics.update(
                    validate_one_epoch(
                        surrogate_model,
                        val_loader,
                        device=device,
                        output_layers=layers,
                        teacher=teacher,
                        state_weight=state_weight,
                        jvp_weight=jvp_weight,
                        kl_weight=kl_weight,
                    )
                )
                if train_eval_loader is not None:
                    train_eval = validate_one_epoch(
                        surrogate_model,
                        train_eval_loader,
                        device=device,
                        output_layers=layers,
                        teacher=teacher,
                        state_weight=state_weight,
                        jvp_weight=jvp_weight,
                        kl_weight=kl_weight,
                        split="train_eval",
                    )
                    metrics.update(train_eval)
                    for name in (
                        "jvp_relative_error",
                        "state_rollout_fvu",
                        "latent_fvu",
                        "kl_divergence",
                    ):
                        if f"val/{name}" in metrics and f"train_eval/{name}" in metrics:
                            metrics[f"gap/{name}"] = (
                                metrics[f"val/{name}"] - metrics[f"train_eval/{name}"]
                            )
            if checkpoint_path is not None and "val/loss" in metrics:
                selection_metric = (
                    "val/jvp_relative_error"
                    if "val/jvp_relative_error" in metrics
                    else "val/loss"
                )
                if metrics[selection_metric] < best_val_metric:
                    best_val_metric = metrics[selection_metric]
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = checkpoint_path.with_name(
                        checkpoint_path.name + ".tmp"
                    )
                    torch.save(
                        {
                            "model_state_dict": surrogate_model.state_dict(),
                            "model_config": {
                                "hidden_width": hidden_width,
                                "latent_width": latent_width,
                                "operator_steps": operator_steps,
                                "operator": operator_name,
                                "flow_layers": len(surrogate_model.flow),
                                "flow_hidden_width": (
                                    surrogate_model.flow[0].net[0].out_features
                                ),
                            },
                            "epoch": epoch + 1,
                            "selection_metric": selection_metric,
                            "validation_metric": best_val_metric,
                        },
                        temporary_path,
                    )
                    temporary_path.replace(checkpoint_path)
                    if run is not None:
                        run.summary["best_epoch"] = epoch + 1
                        run.summary["best_validation_metric"] = best_val_metric
                    print(
                        f"checkpoint epoch={epoch + 1} "
                        f"{selection_metric}={best_val_metric:.6g} "
                        f"path={checkpoint_path}"
                    )
            if run is not None:
                run.log({"epoch": epoch + 1, **metrics})
            print(
                f"epoch {epoch + 1}/{epochs} train_loss={metrics['train/loss']:.6g}"
                + (f" val_loss={metrics['val/loss']:.6g}" if "val/loss" in metrics else "")
            )
    finally:
        if run is not None:
            run.finish()

    return surrogate_model


def load_training_data(
    cache_dir: Path, manifest: dict, *, split: str = "train", include_jvp: bool
) -> TensorDataset:
    """Load aligned raw state rollouts and optional JVPs from one split."""
    layers = manifest["cached_layers"]
    if len(layers) < 2:
        raise ValueError("Training needs at least two cached layers")

    split_info = manifest["splits"][split]
    num_docs = split_info["documents"]
    width = manifest["hidden_width"]
    steps = len(layers) - 1
    inputs = torch.empty((num_docs, width), dtype=torch.float32)
    targets = torch.empty((num_docs, steps, width), dtype=torch.float32)
    if include_jvp:
        if manifest["jvp"] is None:
            raise ValueError("JVP diagnostics require cached directions and responses")
        direction_count = manifest["jvp"]["directions_per_document"]
        directions = torch.empty((num_docs, direction_count, width), dtype=torch.float32)
        responses = torch.empty((num_docs, direction_count, steps, width), dtype=torch.float32)

    offset = 0
    for shard_name in split_info["shards"]:
        states = torch.load(cache_dir / split / shard_name, map_location="cpu", weights_only=True)[
            "activations"
        ]
        count = states.shape[0]
        next_offset = offset + count
        if states.shape[1:] != (len(layers), width) or next_offset > num_docs:
            raise ValueError(f"Unexpected activation shape in {split}/{shard_name}")
        inputs[offset:next_offset].copy_(states[:, 0])
        targets[offset:next_offset].copy_(states[:, 1:])
        if include_jvp:
            jvp = torch.load(
                cache_dir / "jvp" / split / shard_name,
                map_location="cpu",
                weights_only=True,
            )
            if jvp["directions"].shape != (direction_count, count, width) or jvp[
                "responses"
            ].shape != (direction_count, count, steps, width):
                raise ValueError(f"Unexpected JVP shape in jvp/{split}/{shard_name}")
            directions[offset:next_offset].copy_(jvp["directions"].movedim(0, 1))
            responses[offset:next_offset].copy_(jvp["responses"].movedim(0, 1))
        offset = next_offset
    if offset != num_docs:
        raise ValueError(f"Cache manifest expects {num_docs} {split} documents; found {offset}")
    if include_jvp:
        return TensorDataset(inputs, targets, directions, responses)
    return TensorDataset(inputs, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--latent_width", type=int, required=True)
    parser.add_argument(
        "--operator", choices=("shared", "layerwise", "shared-modes"), default="shared"
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak LR after warmup")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--jvp-weight", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log to the tokae W&B project")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.validate_every < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.state_weight < 0:
        raise ValueError("learning_rate must be positive and state-weight nonnegative")
    if args.jvp_weight is not None and args.jvp_weight < 0:
        raise ValueError("jvp-weight must be nonnegative")
    if args.kl_weight < 0:
        raise ValueError("kl-weight must be nonnegative")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if (args.kl_weight or args.wandb) and (
        manifest["cached_layers"][-1] != manifest["num_transformer_layers"]
    ):
        raise ValueError(
            "Teacher-head KL requires the cached endpoint to be the final "
            "transformer layer; cache the last blocks instead"
        )
    jvp_info = manifest["jvp"]
    jvp_weight = (
        (1.0 if jvp_info is not None else 0.0)
        if args.jvp_weight is None
        else args.jvp_weight
    )
    if jvp_weight and jvp_info is None:
        raise ValueError("JVP loss needs cached directions and responses")
    if args.state_weight + jvp_weight + args.kl_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    include_jvp = jvp_info is not None
    train_data = load_training_data(args.cache_dir, manifest, include_jvp=include_jvp)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = None
    if manifest["splits"]["val"]["documents"]:
        val_data = load_training_data(
            args.cache_dir, manifest, split="val", include_jvp=include_jvp
        )
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    train(
        hidden_width=manifest["hidden_width"],
        latent_width=args.latent_width,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        validate_every=args.validate_every,
        learning_rate=args.learning_rate,
        device=device,
        operator_steps=len(manifest["cached_layers"]) - 1,
        operator_name=args.operator,
        state_weight=args.state_weight,
        wandb_logging=args.wandb,
        jvp_weight=jvp_weight,
        kl_weight=args.kl_weight,
        output_layers=manifest["cached_layers"][1:],
        checkpoint_path=args.checkpoint_path,
        run_config={
            "batch_size": args.batch_size,
            "seed": args.seed,
            "teacher_model": manifest["model"],
            "teacher_model_revision": manifest["model_revision"],
            "num_train_docs": manifest["splits"]["train"]["documents"],
            "num_val_docs": manifest["splits"]["val"]["documents"],
            "cached_jvp_directions": 0 if jvp_info is None else jvp_info["directions_per_document"],
            "jvp_directions_used": 0 if jvp_info is None else jvp_info["directions_per_document"],
            "jvp_directions_optimized": 0 if jvp_weight == 0 else jvp_info["directions_per_document"],
            "cached_layers": manifest["cached_layers"],
            "latent_width": args.latent_width,
        },
    )

if __name__ == "__main__":
    main()