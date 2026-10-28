"""Own one Gemma teacher, one persistent training stream, and fixed validation."""

from contextlib import ExitStack
import json
import math
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader

from kae.checkpoints import build_student
from kae.data.cache import ActivationCache
from kae.data.datasets import attach_context
from kae.data.streaming import StreamingActivations
from kae.progress import TrainingProgress
from kae.teacher import GemmaTeacher
from kae.training.checkpointing import PauseRequest, load_resume
from kae.training.config import TrainingConfig
from kae.training.fit import fit
from kae.training.objectives import objective_weights
from kae.wandb import init_run, log_epoch


class Trainer:
    def __init__(self, config: TrainingConfig):
        config.validate()
        self.config = config
        self.paused = False

    def run(self, *, resume=False):
        saved = load_resume(self.config) if resume else None
        self.paused = False
        with PauseRequest(self.config.output_dir) as pause:
            model = self._run(saved, pause)
            self.paused = pause.paused
            return model

    def _run(self, resume, pause):
        config = self.config
        output = Path(config.output_dir)
        if resume is None:
            output.mkdir(parents=True, exist_ok=False)
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resume is not None and resume["completed_epoch"] == config.epochs:
            return self._completed_run(resume, device)

        reporter = TrainingProgress(config.progress_every_seconds)
        cache = ActivationCache(config.validation_cache)
        manifest = cache.manifest
        cache_metadata = cache.metadata()
        if resume is not None and cache_metadata != resume["run_config"]["validation_cache"]:
            raise ValueError("Resume validation cache provenance mismatch")
        if not manifest["splits"]["val"]["documents"]:
            raise ValueError("Training requires a nonempty fixed validation cache")
        if manifest["cached_layers"][-1] != manifest["num_transformer_layers"]:
            raise ValueError("KL evaluation requires the final transformer block boundary")
        if manifest.get("transformers_version") != transformers.__version__:
            raise ValueError("Validation cache and training must use the same Transformers version")
        if manifest.get("layer_indexing") != (
            "1-based raw transformer-block outputs; final model norm is not applied"
        ) or manifest.get("saved_tensor_dtypes", {}).get("activations") != "torch.float32":
            raise ValueError("Validation cache must contain FP32 raw pre-final-norm block states")
        if len(manifest["cached_layers"]) < 3 and (
            config.multistart_state_weight or config.multistart_latent_weight
        ):
            raise ValueError("Multi-start losses need at least two output boundaries")

        metadata = {
            **cache_metadata, **config.as_dict(),
            "actual_device": str(device),
            "num_train_docs": config.docs_per_epoch,
            "num_train_eval_docs": min(256, config.docs_per_epoch),
            "total_training_document_budget": config.epochs * config.docs_per_epoch,
            "total_steps": config.epochs * math.ceil(config.docs_per_epoch / config.batch_size),
            "loss_weights": objective_weights(config),
            "validation_cache": cache_metadata,
        }
        if resume is None:
            (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run = None
        succeeded = False
        try:
            if config.wandb:
                run = init_run(metadata, resume_id=resume["wandb_run"]["id"] if resume else None)
                reporter.attach_run(run)
                identity = {"id": run.id, "url": run.url, "name": run.name}
                (output / "wandb_run.json").write_text(json.dumps(identity, indent=2) + "\n")
            # Close the source and teacher even when extraction or optimization fails.
            with ExitStack() as resources:
                teacher = reporter.call(
                    "teacher_load", GemmaTeacher, cache_metadata["teacher_model"], device,
                    revision=cache_metadata["teacher_model_revision"],
                    layers=manifest["cached_layers"], max_length=manifest["max_length"],
                    prefix_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[config.teacher_precision],
                    collect_context=bool(config.context_width),
                )
                resources.callback(teacher.close)
                if (teacher.hidden_width != manifest["hidden_width"]
                        or teacher.num_layers != manifest["num_transformer_layers"]):
                    raise ValueError("Teacher dimensions disagree with the validation cache")
                val = reporter.call("validation_cache_load", cache.load_split, "val")
                if config.context_width:
                    val = attach_context(val, teacher, cache.token_records("val"),
                                         batch_size=config.teacher_batch_size, reporter=reporter)
                # Logical epochs continue this source; they never reopen the dataset.
                stream = StreamingActivations(
                    cache, teacher, dataset=config.dataset, dataset_config=config.dataset_config,
                    dataset_split=config.dataset_split, dataset_revision=config.dataset_revision,
                    data_files=config.data_files, text_column=config.text_column,
                    docs_per_epoch=config.docs_per_epoch, seed=config.seed,
                    teacher_batch_size=config.teacher_batch_size, reporter=reporter,
                    min_truncation=config.min_truncation, crop_seed=config.seed,
                    resume_state=resume["stream_state_dict"] if resume is not None else None,
                )
                resources.callback(stream.close)
                train_loader = DataLoader(stream, batch_size=config.batch_size, num_workers=0,
                                          generator=torch.Generator().manual_seed(config.seed))
                loaders = {name: DataLoader(data, batch_size=config.batch_size)
                           for name, data in (("train_probe", stream.probe_dataset), ("val", val))}
                metadata.update({**teacher.metadata, **stream.run_config})
                torch.manual_seed(config.seed)
                model = fit(config, cache, train_loader, loaders, teacher.head, metadata,
                            reporter, device, run=run, resume=resume, pause=pause)
            succeeded = True
            return model
        finally:
            try:
                phase = "paused" if succeeded and pause.paused else ("complete" if succeeded else "failed")
                reporter.report(force=True, phase=phase)
            finally:
                if run is not None:
                    run.finish(exit_code=0 if succeeded else 1)

    def _completed_run(self, saved, device):
        """An already-finished resume only restores weights and reconciles W&B."""
        model = build_student(saved["model_config"])
        state = saved["model_state_dict"]
        model.set_standardization(state["mu"], state["sigma"])
        model.load_state_dict(state, strict=True)
        if self.config.wandb:
            run = init_run(saved["run_config"], resume_id=saved["wandb_run"]["id"])
            succeeded = False
            try:
                for row in saved["epoch_history"]:
                    log_epoch(run, row["epoch"], row["metrics"])
                succeeded = True
            finally:
                run.finish(exit_code=0 if succeeded else 1)
        return model.to(device)
