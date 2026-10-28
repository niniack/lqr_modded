"""Student setup, optimizer epochs, and the training/validation lifecycle."""

from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from kae.checkpoints import build_student
from kae.data.normalization import fit_layerwise_standardization
from kae.training.optimization import Optimization
from kae.training.checkpointing import CheckpointWriter, load_initial_weights, restore_rng
from kae.training.diagnostics import fixed_probe_gradients
from kae.training.evaluation import evaluate_clean
from kae.training.objectives import compute_objectives, objective_weights, to_device
from kae.wandb import log_epoch, run_metadata


def fit(config, cache, train_loader, loaders, teacher_head, metadata, reporter, device,
        *, run=None, resume=None, pause=None):
    layers = cache.manifest["cached_layers"][1:]
    model_config = {
        "architecture": config.architecture,
        "hidden_width": cache.manifest["hidden_width"],
        "latent_width": config.latent_width or cache.manifest["hidden_width"],
        "operator_steps": len(layers), "operator": config.operator,
        "mlp_width": config.mlp_width, "student_normalization": config.student_normalization,
        "normalization_fit": {
            "source": ("fixed pre-optimization training probe"
                       if config.student_normalization == "layerwise" else "constructor identity"),
            "fitted_training_documents": (len(loaders["train_probe"].dataset)
                                          if config.student_normalization == "layerwise" else 0),
            "boundaries": list(range(len(layers) + 1)),
            "teacher_boundaries": list(cache.manifest["cached_layers"]),
            "moments": ("CPU FP64 population mean/std (correction=0), per boundary and feature"
                        if config.student_normalization == "layerwise" else "mu=0, sigma=1; no fit"),
            "sigma_floor_rule": ("per boundary max(1e-3 * median(feature population std), 1e-6)"
                                 if config.student_normalization == "layerwise" else "not applied"),
        },
    }
    if config.architecture == "aikae":
        model_config.update(flow_layers=config.flow_layers,
                            flow_hidden_width=config.flow_hidden_width)
    if config.context_width:
        model_config.update(context_width=config.context_width,
                            context_tokens=cache.manifest["max_length"] - 1)
    if resume is not None:
        model_config = resume["model_config"]
    model = build_student(model_config).to(device)
    if resume is not None:
        state = resume["model_state_dict"]
        model.set_standardization(state["mu"], state["sigma"])
        model.load_state_dict(state, strict=True)
    elif config.initial_checkpoint:
        metadata["initialization"] = load_initial_weights(
            model, model_config, config.initial_checkpoint, cache.metadata())
        model_config["normalization_fit"] = metadata["initialization"]["normalization_fit"]
    elif config.student_normalization == "layerwise":
        mu, sigma = fit_layerwise_standardization(loaders["train_probe"].dataset.tensors)
        model.set_standardization(mu, sigma)
    # The head is frozen, but KL gradients still flow through it to student states.
    teacher_head.eval().requires_grad_(False)
    objective = dict(teacher=teacher_head, weights=objective_weights(config))
    total_steps = config.epochs * len(train_loader)
    optimization = Optimization(model, config, total_steps)
    metadata = (resume["run_config"] if resume is not None else run_metadata(
        config, metadata, model, model_config, train_loader, reporter,
        total_steps, optimization.warmup_steps, loss_weights=objective["weights"]))
    output = Path(config.output_dir)
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if run is not None:
        run.config.update(metadata, allow_val_change=True)
    checkpoints = CheckpointWriter(config.output_dir, model_config, metadata, reporter,
                                   resume=resume)
    evaluation = dict(device=device, output_layers=layers)
    # This fixed training batch is never sampled from the optimizer iterator.
    with torch.random.fork_rng(devices=[]):
        gradient_batch = to_device(next(iter(loaders["train_probe"])), device)
    if resume is not None:
        optimization.load_state_dict(resume["optimization_state_dict"])
        train_loader.generator.set_state(resume["train_loader_generator_state"])
        reporter.load_state_dict(resume["progress_state"])
        if run is not None:
            reporter.attach_run(run)
            for row in resume["epoch_history"]:
                log_epoch(run, row["epoch"], row["metrics"])
        # Restore last, after all setup, loader construction, and remote-log replay.
        restore_rng(resume["rng_state"])
    reporter.report(force=True, phase="initializing_training",
                    total_optimizer_steps=total_steps,
                    total_training_documents=len(train_loader.dataset) * config.epochs,
                    epoch_document_budget=len(train_loader.dataset),
                    student_batch_size=config.batch_size,
                    teacher_batch_size=config.teacher_batch_size)
    # Epoch 0 measures initialization; subsequent epochs consume new streamed prompts.
    first_epoch = resume["completed_epoch"] + 1 if resume is not None else 0
    for epoch in range(first_epoch, config.epochs + 1):
        reporter.report(force=True, phase="epoch_start", epoch=epoch, epochs=config.epochs,
                        epoch_documents=0)
        sample_step = max(0, epoch - 1) * len(train_loader)
        metrics = fixed_probe_gradients(model, gradient_batch, objective=objective)
        metrics["train/grad_sample_optimizer_step"] = sample_step
        if epoch:
            metrics.update(train_one_epoch(
                model, train_loader, optimization, device=device, objective=objective,
                reporter=reporter))
        metrics["optimizer_step"] = epoch * len(train_loader)
        metrics.setdefault("train/learning_rate", optimization.learning_rate)
        for group in optimization.param_groups:
            metrics.setdefault(f"train/learning_rate/{group['name']}", float(group["lr"]))
        # A requested pause validates even an epoch outside the normal cadence.
        validated = (epoch == 0 or epoch % config.validate_every == 0
                     or epoch == config.epochs or (pause is not None and pause.requested))
        if validated:
            with torch.random.fork_rng(devices=[]):
                metrics.update(reporter.call(
                    "train_probe_evaluation", evaluate_clean, model, loaders["train_probe"],
                    **evaluation, objective=objective, split="train"))
                metrics.update(reporter.call(
                    "validation", evaluate_clean, model, loaders["val"],
                    **evaluation, objective=objective, split="val"))
        metrics.update(train_loader.dataset.progress())
        checkpoints.record(model, epoch, metrics, final=epoch == config.epochs, run=run)
        if run is not None:
            log_epoch(run, epoch, metrics)
        if validated:
            print(f"epoch {epoch}/{config.epochs}"
                  f" | train loss {metrics['train/loss']:.5g}"
                  f" | val loss {metrics['val/loss']:.5g}"
                  f" | val state FVU {metrics['val/state_rollout_fvu']:.4g}"
                  f" | val latent FVU {metrics['val/latent_fvu']:.4g}"
                  f" | val KL {metrics['val/kl_divergence']:.4g}", flush=True)
        reporter.report(force=True, phase="epoch_complete", epoch=epoch)
        checkpoints.save_resume(model, epoch, metrics, config=config,
                                train_loader=train_loader, optimization=optimization, run=run)
        if (pause is not None and pause.requested and validated
                and epoch < config.epochs):
            pause.pause()
            reporter.report(force=True, phase="paused", epoch=epoch)
            break
    return model


def train_one_epoch(model, train_loader, optimization, *, device, objective, reporter=None):
    model.train()
    totals = defaultdict(float)
    examples = updates = 0
    batches = reporter.batches(train_loader) if reporter is not None else train_loader
    for source in batches:
        started = time.monotonic()
        batch = to_device(source, device)
        # Teacher targets are fixed: backpropagate only through the student's rollout.
        optimization.zero_grad()
        result = compute_objectives(model, batch, **objective)
        result.loss.backward()
        update_stats = optimization.step()
        totals["grad_norm_pre_clip"] += update_stats["grad_norm_pre_clip"]
        learning_rate = update_stats["learning_rate"]
        count = batch["inputs"].shape[0]
        examples += count
        updates += 1
        totals["update_loss"] += result.loss.detach().item() * count
        for name, term in result.terms.items():
            totals[f"update_objective/{name}"] += term.detach().item() * count
        if reporter is not None:
            reporter.record_update(time.monotonic() - started, count)
            reporter.report(phase="training", epoch_documents=examples,
                            batch_loss=result.loss.detach().item(), learning_rate=learning_rate,
                            **train_loader.dataset.progress())
    if not updates:
        raise ValueError("train_loader must yield at least one batch")
    return {**{f"train/{name}": value for name, value in update_stats.items()
               if name.startswith("learning_rate")},
            **{f"train/{name}": value / (updates if name == "grad_norm_pre_clip" else examples)
               for name, value in totals.items()}}
