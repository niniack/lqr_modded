"""Deployment checkpoints, completed-epoch resume, warmstarts, and safe pause requests."""

from pathlib import Path
import random
import signal

import numpy as np
import torch

from kae.checkpoints import load_student, save_checkpoint


RESUME_VERSION = 2
PAUSED_EXIT_CODE = 75


def load_resume(config):
    """Require a full checkpoint and the exact recipe, never a deployment warmstart."""
    path = Path(config.output_dir) / "resume.pt"
    if not path.is_file():
        raise FileNotFoundError(f"--resume requires a full training checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model_state_dict", "model_config", "run_config", "recipe", "completed_epoch",
        "optimization_state_dict", "stream_state_dict", "train_loader_generator_state",
        "rng_state", "progress_state", "best_metric", "wandb_run", "metrics", "epoch_history",
    }
    if (not isinstance(saved, dict) or saved.get("resume_version") != RESUME_VERSION
            or saved.get("checkpoint_version") != 2
            or saved.get("checkpoint_kind") != "resume" or not required.issubset(saved)):
        raise ValueError(f"Not a complete supported training resume checkpoint: {path}")
    recipe = config.as_dict()
    saved_recipe = {"architecture": "aikae", "reconstruction_weight": 0.0, **saved["recipe"]}
    if saved_recipe != recipe:
        changed = sorted(key for key in saved_recipe.keys() | recipe.keys()
                         if saved_recipe.get(key) != recipe.get(key))
        raise ValueError(f"Resume recipe mismatch in: {', '.join(changed)}")
    epoch = saved["completed_epoch"]
    if type(epoch) is not int or not 0 <= epoch <= config.epochs:
        raise ValueError("Resume completed_epoch is outside the configured training target")
    if [row["epoch"] for row in saved["epoch_history"]] != list(range(epoch + 1)):
        raise ValueError("Resume epoch history is not contiguous through completed_epoch")
    steps_per_epoch = (config.docs_per_epoch + config.batch_size - 1) // config.batch_size
    if saved["optimization_state_dict"].get("update_count") != epoch * steps_per_epoch:
        raise ValueError("Resume optimizer update_count does not match its completed epoch")
    identity = saved["wandb_run"]
    if config.wandb and (not isinstance(identity, dict) or not identity.get("id")):
        raise ValueError("Resume checkpoint has no W&B run identity")
    return saved


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    cuda = state["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Resume requires the same number of visible CUDA devices as its checkpoint")
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["algorithm"], np.asarray(numpy["keys"], dtype=np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


class PauseRequest:
    """SIGUSR1 only flips a flag; file requests are checked at completed boundaries."""

    def __init__(self, output_dir):
        self.path = Path(output_dir) / "pause.request"
        self.signaled = False
        self.paused = False
        self._previous_handler = None

    def _request(self, signum, frame):
        self.signaled = True

    @property
    def requested(self):
        return self.signaled or self.path.exists()

    def pause(self):
        self.paused = True
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        self._previous_handler = signal.signal(signal.SIGUSR1, self._request)
        return self

    def __exit__(self, exc_type, exc, traceback):
        signal.signal(signal.SIGUSR1, self._previous_handler)


class CheckpointWriter:
    def __init__(self, output_dir, model_config, run_config, reporter, *, resume=None):
        self.directory = Path(output_dir)
        self.model_config = model_config
        self.run_config = run_config
        self.reporter = reporter
        self.best_metric = resume["best_metric"] if resume is not None else float("inf")
        self.epoch_history = list(resume["epoch_history"]) if resume is not None else []

    def record(self, model, epoch, metrics, *, final, run):
        selection = "val/loss"
        value = metrics.get(selection)
        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": self.model_config,
            "run_config": self.run_config,
            "epoch": epoch,
            "metrics": metrics,
            "selection_metric": selection,
            "validation_metric": value,
        }
        if value is not None and value < self.best_metric:
            self.best_metric = value
            self.reporter.call("checkpoint_write", save_checkpoint,
                               self.directory / "best.pt", {**payload, "checkpoint_kind": "best"})
            if run is not None:
                run.summary["best_epoch"] = epoch
                run.summary["best_validation_metric"] = value
                run.summary["selection_metric"] = selection
            print(f"checkpoint epoch={epoch} {selection}={value:.6g} path={self.directory / 'best.pt'}")
        if final:
            self.reporter.call("checkpoint_write", save_checkpoint,
                               self.directory / "final.pt", {**payload, "checkpoint_kind": "final"})

    def save_resume(self, model, epoch, metrics, *, config, train_loader, optimization, run):
        """Commit the latest completed boundary independently of deployment selection."""
        self.epoch_history.append({"epoch": epoch, "metrics": metrics})
        payload = {
            "checkpoint_kind": "resume",
            "resume_version": RESUME_VERSION,
            "recipe": config.as_dict(),
            "model_state_dict": model.state_dict(),
            "model_config": self.model_config,
            "run_config": self.run_config,
            "epoch": epoch,
            "completed_epoch": epoch,
            "metrics": metrics,
            "epoch_history": self.epoch_history,
            "best_metric": self.best_metric,
            "optimization_state_dict": optimization.state_dict(),
            "stream_state_dict": train_loader.dataset.state_dict(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "progress_state": self.reporter.state_dict(),
            "wandb_run": ({"id": run.id, "url": run.url, "name": run.name}
                          if run is not None else None),
            "rng_state": capture_rng(),
        }
        # Do not time this write inside the state being written: that would make
        # resumed progress omit a duration included by uninterrupted execution.
        save_checkpoint(self.directory / "resume.pt", payload)


def load_initial_weights(model, config, checkpoint, teacher):
    """Load compatible model weights into a fresh optimizer and schedule."""
    source, metadata = load_student(checkpoint, "cpu")
    source_config = metadata["model_config"]
    architecture = config.get("architecture", "aikae")
    if source_config.get("architecture", "aikae") != architecture:
        raise ValueError("Initial checkpoint uses a different encoder/decoder architecture")
    keys = ("hidden_width", "latent_width", "operator", "operator_steps",
            "mlp_width", "student_normalization", "context_width", "context_tokens")
    if architecture == "aikae":
        keys += ("flow_layers", "flow_hidden_width")
    for key in keys:
        default = 0 if key.startswith("context_") else None
        if source_config.get(key, default) != config.get(key, default):
            raise ValueError(f"Initial checkpoint architecture differs at {key}")
    for key, expected in (("teacher_model", teacher["teacher_model"]),
                          ("teacher_revision", teacher["teacher_model_revision"])):
        if metadata[key] != expected:
            raise ValueError(f"Initial checkpoint differs at {key}")
    if (metadata["deployment"]["input_layer"] != teacher["input_layer"]
            or metadata["deployment"]["output_layers"] != teacher["output_layers"]):
        raise ValueError("Initial checkpoint has different teacher boundaries")
    model.set_standardization(source.mu, source.sigma)
    model.load_state_dict(source.state_dict(), strict=True)
    return {
        "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "epoch": metadata["epoch"],
        "normalization_fit": source_config.get("normalization_fit", {"source": "initial checkpoint"}),
        "optimizer_and_schedule": "fresh; not resumed",
    }
