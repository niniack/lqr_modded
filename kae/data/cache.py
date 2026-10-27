"""Read and write fixed raw-state caches with exact tokens and teacher metadata."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch
from .datasets import DictionaryDataset
from .tokens import token_hash

SHARD_SIZE = 256


class ActivationCache:
    """Read existing raw-coordinate caches without consuming cached training data."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        layers = self.manifest["cached_layers"]
        if len(layers) < 2 or layers != list(range(layers[0], layers[-1] + 1)):
            raise ValueError("Cached boundaries must be contiguous raw residual layers")

    def metadata(self) -> dict:
        return _cache_metadata(self.directory, self.manifest)

    def load_split(self, split: str) -> DictionaryDataset:
        if split not in ("val", "test"):
            raise ValueError("Only fixed val/test splits are available; training is streaming")
        return _load_split(self.directory, self.manifest, split=split)

    def token_records(self, split: str) -> list[dict]:
        """Return exact stored unpadded tokens, without loading any derivative cache."""
        if split not in ("val", "test"):
            raise ValueError("Only fixed val/test token records are available")
        return list(_token_records(self.directory, self.manifest, split))

    def heldout_hashes(self) -> set[bytes]:
        return _heldout_hashes(self.directory, self.manifest)


def _cache_metadata(cache_dir: Path, manifest: dict) -> dict:
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
    }
    config.update(
        {f"num_{split}_docs": info["documents"] for split, info in manifest["splits"].items()}
    )
    return config


def _load_split(
    cache_dir: Path, manifest: dict, *, split: str = "val"
) -> DictionaryDataset:
    """Load clean aligned raw rollouts, ignoring historical derivative shards."""
    layers = manifest["cached_layers"]
    if len(layers) < 2:
        raise ValueError("Training needs at least two cached layers")

    split_info = manifest["splits"][split]
    num_docs = split_info["documents"]
    width = manifest["hidden_width"]
    steps = len(layers) - 1
    inputs = torch.empty((num_docs, width), dtype=torch.float32)
    targets = torch.empty((num_docs, steps, width), dtype=torch.float32)

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
        offset = next_offset
    if offset != num_docs:
        raise ValueError(f"Cache manifest expects {num_docs} {split} documents; found {offset}")
    return DictionaryDataset({"inputs": inputs, "targets": targets})


def _token_records(cache_dir: Path, manifest: dict, split: str):
    """Read token storage while memory-mapping, rather than copying, raw states."""
    info = manifest.get("splits", {}).get(split)
    if not isinstance(info, dict) or "shards" not in info:
        raise ValueError(f"Cache manifest must describe {split!r} shards")
    documents = 0
    for shard_name in info["shards"]:
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ValueError(f"Invalid {split} cache shard name: {shard_name!r}")
        path = cache_dir / split / shard_name
        shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        ids = shard.get("input_ids")
        offsets = shard.get("input_offsets")
        if (
            not isinstance(ids, torch.Tensor)
            or not isinstance(offsets, torch.Tensor)
            or ids.dtype != torch.long
            or offsets.dtype != torch.long
            or ids.ndim != 1
            or offsets.ndim != 1
            or offsets.numel() == 0
        ):
            raise ValueError(f"Cache shard needs flat int64 input_ids/input_offsets: {path}")
        boundaries = offsets.tolist()
        if (
            boundaries[0] != 0
            or boundaries[-1] != ids.numel()
            or any(end <= start for start, end in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(f"Invalid token offsets in {path}")
        for start, end in zip(boundaries, boundaries[1:]):
            tokens = ids[start:end].clone()
            yield {"input_ids": tokens, "prompt_id": token_hash(tokens)}
        documents += len(boundaries) - 1
    if documents != info["documents"]:
        raise ValueError(f"{split} shard document count does not match cache manifest")


def _heldout_hashes(cache_dir: Path, manifest: dict) -> set[bytes]:
    return {
        record["prompt_id"]
        for split in ("val", "test")
        for record in _token_records(cache_dir, manifest, split)
    }


def write_split(
    split_name,
    document_count,
    source,
    text_column,
    output_dir,
    teacher,
    batch_size,
):
    if type(document_count) is not int or document_count < 1:
        raise ValueError("document_count must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_dir = Path(output_dir)
    split_dir = output_dir / split_name
    split_dir.mkdir()

    shard_names = []
    pending_activations = []
    pending_input_ids = []
    pending_count = 0
    documents_written = 0
    shard_index = 0
    saved_tensor_dtypes = None

    def save_shard():
        nonlocal pending_count, shard_index, saved_tensor_dtypes
        if not pending_count:
            return
        shard_name = f"{shard_index:06d}.pt"
        shard_activations = torch.cat(pending_activations)
        input_offsets = [0]
        for token_ids in pending_input_ids:
            input_offsets.append(input_offsets[-1] + token_ids.numel())
        shard_input_ids = torch.cat(pending_input_ids)
        shard_input_offsets = torch.tensor(input_offsets, dtype=torch.long)
        torch.save(
            {
                "activations": shard_activations,
                "input_ids": shard_input_ids,
                "input_offsets": shard_input_offsets,
            },
            split_dir / shard_name,
        )
        dtypes = {
            "activations": str(shard_activations.dtype),
            "input_ids": str(shard_input_ids.dtype),
            "input_offsets": str(shard_input_offsets.dtype),
        }
        if saved_tensor_dtypes is None:
            saved_tensor_dtypes = dtypes
        shard_names.append(shard_name)
        pending_activations.clear()
        pending_input_ids.clear()
        pending_count = 0
        shard_index += 1

    while documents_written < document_count:
        batch_count = min(batch_size, document_count - documents_written)
        tokens = []
        for _ in range(batch_count):
            try:
                example = next(source)
            except StopIteration as error:
                raise ValueError(
                    f"Dataset split ended before collecting {document_count} {split_name} documents"
                ) from error
            text = example.get(text_column) if isinstance(example, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Dataset field {text_column!r} must contain non-empty text in {split_name}"
                )
            tokens.append(teacher.tokenize(text))

        targets = teacher.targets_batch(tokens)
        activations = torch.cat(
            (targets["inputs"].unsqueeze(1), targets["targets"]), dim=1
        ).detach().cpu()
        token_ids = [
            token["input_ids"][0][token["attention_mask"][0].bool()].detach().cpu()
            for token in tokens
        ]
        offset = 0
        while offset < batch_count:
            chunk_count = min(SHARD_SIZE - pending_count, batch_count - offset)
            chunk_end = offset + chunk_count
            pending_activations.append(activations[offset:chunk_end])
            pending_input_ids.extend(token_ids[offset:chunk_end])
            pending_count += chunk_count
            offset = chunk_end
            if pending_count == SHARD_SIZE:
                save_shard()
        documents_written += batch_count

    save_shard()
    split_info = {
        "documents": documents_written,
        "shards": shard_names,
        "tensor_dtypes": saved_tensor_dtypes,
    }
    return split_info

