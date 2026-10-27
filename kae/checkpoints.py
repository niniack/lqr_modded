"""Strict student construction, deployment restoration, and atomic checkpoints."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from kae.models import (AIKAE, KoopmanOperator, LayerwiseKoopmanOperator,
                        ResidualLayerwiseKoopmanOperator, SharedModesKoopmanOperator)
from kae.models.context_mixer import ContextMixer


CHECKPOINT_VERSION = 2
RAW_COORDINATES = "raw transformer-block final-token hidden states"


def _context_config(config: dict) -> tuple[int, int]:
    width, tokens = config.get("context_width", 0), config.get("context_tokens", 0)
    if (not isinstance(width, int) or isinstance(width, bool) or width < 0
            or not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
            or (width > 0) != (tokens > 0)):
        raise ValueError("Context widths/tokens must be positive integers together, or both zero")
    return width, tokens


def _validate_state(config: dict, state: dict) -> None:
    """Reject incomplete or non-FP32 affine student states before construction."""
    if not isinstance(state, dict) or not all(isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError("Student model_state_dict must contain only tensors")
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("Student checkpoint contains non-FP32 floating-point tensors")
    width = config["hidden_width"]
    if "mu" not in state or "sigma" not in state:
        raise ValueError("Version 2 checkpoints require explicit mu and sigma buffers")
    mu, sigma = state["mu"], state["sigma"]
    shapes = ((width,), (config["operator_steps"] + 1, width))
    if (mu.dtype != torch.float32 or sigma.dtype != torch.float32
            or mu.shape not in shapes or sigma.shape != mu.shape):
        raise ValueError(
            "Student affine buffers must be matching FP32 [hidden_width] or "
            "[operator_steps + 1, hidden_width]"
        )
    if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0).all()):
        raise ValueError("Student affine buffers must be finite with strictly positive sigma")
    context_width, context_tokens = _context_config(config)
    context_state = {
        key.removeprefix("context_mixer."): tensor
        for key, tensor in state.items() if key.startswith("context_mixer.")
    }
    expected = {}
    if context_width:
        # Validate every mixer key and shape without allocating model weights.
        with torch.device("meta"):
            expected = ContextMixer(
                width, context_width, context_tokens, config["latent_width"]
            ).state_dict()
    if (context_state.keys() != expected.keys()
            or any(context_state[key].shape != tensor.shape for key, tensor in expected.items())):
        raise ValueError("Context mixer state does not match context_width/context_tokens metadata")


def _normalization_metadata(state: dict, config: dict) -> dict:
    mu, sigma = state["mu"], state["sigma"]
    metadata = {
        "external_coordinates": RAW_COORDINATES,
        "external_input_standardization": "none",
        "external_response_normalization": "none",
        "final_teacher_norm_applied": False,
        "internal_preconditioner": {
            "kind": "fixed-affine",
            "encode": "(x - mu) / sigma",
            "decode": "u * sigma + mu",
            "buffers": ["mu", "sigma"],
            "dtype": "torch.float32",
            "identity": bool((mu == 0).all()) and bool((sigma == 1).all()),
        },
        "flow_condition_norm": "non-affine LayerNorm on coupling conditioning half only",
    }
    if mu.ndim == 2:
        metadata["internal_preconditioner"].update({
            "scope": "per-boundary",
            "boundaries": list(range(mu.shape[0])),
            "shape": list(mu.shape),
            "encode": "(x - mu[boundary]) / sigma[boundary]",
            "decode": "u * sigma[boundary] + mu[boundary]",
        })
    if config.get("context_width", 0):
        metadata["context_preconditioner"] = {
            "coordinates": "raw preceding-token residual states at the input boundary",
            "encode": "(context - mu[0]) / sigma[0]" if mu.ndim == 2 else "(context - mu) / sigma",
            "mask": "valid preceding tokens only; excludes final token and padding",
            "pooling": "masked mean; empty prefix gives zero forcing",
        }
    return metadata


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot freeze metadata of type {type(value).__name__}")


def build_student(config: dict) -> AIKAE:
    """Construct all saved architecture dimensions without disturbing caller RNG."""
    operators = {
        "shared": KoopmanOperator,
        "layerwise": LayerwiseKoopmanOperator,
        "residual-layerwise": ResidualLayerwiseKoopmanOperator,
        "shared-modes": SharedModesKoopmanOperator,
    }
    if config["operator"] not in operators:
        raise ValueError(f"Unknown Koopman operator: {config['operator']}")
    context_width, context_tokens = _context_config(config)
    # Operator sizes must not alter the shared encoder's seeded initialization.
    with torch.random.fork_rng(devices=[]):
        operator = operators[config["operator"]](config["latent_width"], steps=config["operator_steps"])
    return AIKAE(hidden_width=config["hidden_width"], latent_width=config["latent_width"],
                 mlp_width=config.get("mlp_width", 1280), flow_layers=config.get("flow_layers", 6),
                 flow_hidden_width=config.get("flow_hidden_width", 768), operator=operator,
                 context_width=context_width, context_tokens=context_tokens)


def load_student(
    checkpoint: str | Path, device: str | torch.device
) -> tuple[AIKAE, dict]:
    """Strictly restore a version 2 FP32 student with raw external coordinates."""
    path = Path(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"checkpoint_version", "model_config", "run_config", "model_state_dict", "normalization"}
    if not isinstance(saved, dict) or saved.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Expected a version 2 AIKAE checkpoint")
    if not required.issubset(saved):
        raise ValueError(f"Incomplete version 2 checkpoint: missing {sorted(required - saved.keys())}")
    if not {"hidden_width", "latent_width", "operator_steps", "operator"}.issubset(saved["model_config"]):
        raise ValueError("Unsupported model_config; expected a modern AIKAE architecture")
    config = saved["model_config"]
    run = saved.get("run_config", {})
    manifest = run.get("cache_manifest", {})
    provenance = run.get("cache_provenance", {})
    teacher_model = (
        run.get("teacher_model") or provenance.get("teacher_model")
        or run.get("teacher_checkpoint_path") or manifest.get("model")
    )
    teacher_revision = (
        run.get("teacher_model_revision") or run.get("teacher_revision")
        or provenance.get("teacher_model_revision") or provenance.get("revision")
        or run.get("teacher_checkpoint_revision") or manifest.get("model_revision")
    )
    layers = run.get("cached_layers", manifest.get("cached_layers"))
    input_layer = run.get("input_layer", layers[0] if layers else None)
    output_layers = run.get("output_layers", layers[1:] if layers else None)
    if not teacher_model or input_layer is None or not output_layers:
        raise ValueError("Checkpoint must identify its teacher and deployment boundaries")
    if output_layers != list(range(input_layer + 1, input_layer + 1 + config["operator_steps"])):
        raise ValueError("Checkpoint deployment boundaries do not match operator steps")
    indexing = manifest.get("layer_indexing")
    if indexing is not None and indexing != "1-based raw transformer-block outputs; final model norm is not applied":
        raise ValueError("Checkpoint must use raw pre-final-norm residual coordinates")
    state = saved["model_state_dict"]
    _validate_state(config, state)
    if saved["normalization"] != _normalization_metadata(state, config):
        raise ValueError("Checkpoint normalization metadata does not match its affine student buffers")
    with torch.random.fork_rng(devices=[]):
        student = build_student(config)
    student.set_standardization(state["mu"], state["sigma"])
    student.load_state_dict(state, strict=True)
    student.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    metadata = {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": saved.get("checkpoint_kind", "best"),
        "epoch": saved.get("epoch"),
        "model_config": config,
        "run_config": run,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "strict_state_dict": True,
        "parameter_count": sum(parameter.numel() for parameter in student.parameters()),
        "dtype": "torch.float32",
        "device": str(torch.device(device)),
        "normalization": saved["normalization"],
        "deployment": {
            "input_layer": input_layer,
            "output_layers": output_layers,
            "encode_calls_per_rollout": 1,
            "propagation": "pure latent; decode each transition; no decoded-state feedback",
            "model_class": "kae.models.AIKAE",
        },
    }
    if config.get("context_width", 0):
        metadata["deployment"]["context_conditioning"] = {
            "kind": "two-block masked MLP-Mixer",
            "context_width": config["context_width"],
            "context_tokens": config["context_tokens"],
            "forcing": "z1 = K1(z0) + u; subsequent transitions are pure latent",
        }
    return student, _json_safe(metadata)


def save_checkpoint(path: Path, payload: dict, *, overwrite: bool = True) -> None:
    """Publish a complete version 2 checkpoint atomically in its destination directory."""
    if payload.get("checkpoint_version", CHECKPOINT_VERSION) != CHECKPOINT_VERSION:
        raise ValueError("Only version 2 student checkpoints may be saved")
    required = {"model_config", "run_config", "model_state_dict"}
    if not required.issubset(payload):
        raise ValueError(f"Incomplete checkpoint payload: missing {sorted(required - payload.keys())}")
    _validate_state(payload["model_config"], payload["model_state_dict"])
    payload = {
        **payload,
        "checkpoint_version": CHECKPOINT_VERSION,
        "normalization": _normalization_metadata(payload["model_state_dict"], payload["model_config"]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            temporary_path.replace(path)
        else:
            path.hardlink_to(temporary_path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)

