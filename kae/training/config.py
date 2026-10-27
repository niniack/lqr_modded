"""One flat TOML recipe for clean, streaming Gemma training."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib


@dataclass
class TrainingConfig:
    # The cache fixes the teacher, suffix boundaries, and held-out documents.
    validation_cache: str = "data/activations/gemma3-4b"
    output_dir: str = "runs/gemma3-4b"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = None
    dataset_split: str = "train"
    dataset_revision: str | None = None
    data_files: list[str] | str | None = None
    text_column: str = "text"
    docs_per_epoch: int = 32768
    epochs: int = 10
    batch_size: int = 32
    teacher_batch_size: int = 8
    min_truncation: int = 2
    teacher_precision: str = "bfloat16"

    # None means no augmentation: latent width equals the teacher's hidden width.
    latent_width: int | None = None
    flow_layers: int = 6
    flow_hidden_width: int = 768
    mlp_width: int = 1280
    operator: str = "layerwise"
    student_normalization: str = "none"
    context_width: int = 0
    initial_checkpoint: str | None = None

    # Muon for matrix weights, Adam for biases; linear warmup then cosine decay.
    learning_rate: float = 1e-4
    muon_lr_multiplier: float = 1.5
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    state_weight: float = 0.001
    latent_weight: float = 0.0
    kl_weight: float = 2.5
    multistart_state_weight: float = 0.0
    multistart_latent_weight: float = 0.0

    validate_every: int = 1
    seed: int = 1729
    device: str | None = None
    wandb: bool = True
    wandb_base_url: str | None = "https://api.wandb.ai"
    wandb_entity: str | None = "nishantaswani"
    run_name: str | None = None
    progress_every_seconds: float = 60.0

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("rb") as handle:
            return cls(**tomllib.load(handle))

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for name in ("docs_per_epoch", "epochs", "batch_size", "teacher_batch_size",
                     "validate_every", "flow_layers", "flow_hidden_width", "mlp_width",
                     "min_truncation"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.latent_width is not None and (type(self.latent_width) is not int or self.latent_width < 1):
            raise ValueError("latent_width must be a positive integer or omitted")
        if type(self.context_width) is not int or self.context_width < 0:
            raise ValueError("context_width must be a nonnegative integer; 0 disables context")
        if not self.dataset:
            raise ValueError("A streaming dataset is required")
        for name in ("learning_rate", "muon_lr_multiplier", "warmup_fraction", "grad_clip",
                     "state_weight", "latent_weight", "kl_weight", "multistart_state_weight",
                     "multistart_latent_weight", "progress_every_seconds"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.muon_lr_multiplier <= 0 or self.grad_clip <= 0:
            raise ValueError("Learning rate, Muon multiplier, and gradient clip must be positive")
        if not 0 < self.warmup_fraction <= 1 or self.progress_every_seconds <= 0:
            raise ValueError("Require 0 < warmup_fraction <= 1 and a positive progress interval")
        weights = (self.state_weight, self.latent_weight, self.kl_weight,
                   self.multistart_state_weight, self.multistart_latent_weight)
        if min(weights) < 0 or sum(weights) == 0:
            raise ValueError("Loss weights must be nonnegative and not all zero")
        if self.teacher_precision not in ("float32", "bfloat16"):
            raise ValueError("teacher_precision must be float32 or bfloat16")
        if self.operator not in ("shared", "layerwise", "residual-layerwise", "shared-modes"):
            raise ValueError("Unknown Koopman operator")
        if self.student_normalization not in ("none", "layerwise"):
            raise ValueError("student_normalization must be none or layerwise")
