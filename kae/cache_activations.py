"""Prepare disjoint clean validation/test raw states and exact token IDs.

Training consumes a live stream, not these shards. Each fixed split uses
256-document shards and retains unpadded token IDs alongside its targets.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import transformers
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
from huggingface_hub import HfApi

from .data.cache import SHARD_SIZE, write_split
from .data.prompts import SHUFFLE_BUFFER_SIZE, load_prompt_stream
from .data.tokens import token_hash
from .teacher import GemmaTeacher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face Gemma ID or local model path")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset ID or builder, e.g. json")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--data-files", nargs="+", help="Optional files for dataset builders such as json")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--val-docs", type=int, required=True)
    parser.add_argument("--test-docs", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--layers", type=int, nargs=2, metavar=("START", "END"),
        help="Inclusive 1-based raw block-output boundaries (at least two)",
    )
    selection.add_argument(
        "--last-layers", type=int, metavar="STEPS",
        help="Cache the input and outputs of the final STEPS blocks (default 4)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", help="Device such as cuda, cuda:0, or cpu")
    return parser


def _dataset_revision(args):
    """Resolve Hub datasets before opening the stream; local builders need no pin."""
    if args.dataset in _PACKAGED_DATASETS_MODULES or Path(args.dataset).exists():
        return args.dataset_revision
    return HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha


def _local_file_hashes(data_files):
    """Record exact bytes for local builder inputs, alongside their original paths."""
    hashes = {}
    for name in data_files or []:
        path = Path(name)
        if path.is_file():
            with path.open("rb") as source:
                hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return hashes


def _distinct_documents(source, teacher, text_column):
    """Keep capped token identities unique across the shared val/test source."""
    seen = set()
    for example in source:
        text = example.get(text_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset field {text_column!r} must contain non-empty text")
        identity = token_hash(teacher.tokenize(text)["input_ids"][0])
        if identity not in seen:
            seen.add(identity)
            yield example


def main() -> None:
    args = _parser().parse_args()
    if min(args.val_docs, args.test_docs) < 0:
        raise ValueError("Document counts cannot be negative")
    if args.val_docs + args.test_docs < 1:
        raise ValueError("Prepare at least one fixed validation or test document")
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")
    if args.layers is not None and args.layers[0] >= args.layers[1]:
        raise ValueError("layers must be an inclusive contiguous range with START < END")
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    layers = None if args.layers is None else list(range(args.layers[0], args.layers[1] + 1))
    teacher = GemmaTeacher(
        args.model, device, revision=args.model_revision, layers=layers,
        last_layers=args.last_layers, max_length=args.max_length,
    )
    dataset = source = distinct_source = None
    try:
        dataset_revision = _dataset_revision(args)
        data_file_hashes = _local_file_hashes(args.data_files)
        dataset = load_prompt_stream(
            args.dataset, args.dataset_config, split=args.dataset_split,
            data_files=args.data_files, revision=dataset_revision,
        )
        count = args.val_docs + args.test_docs
        source = iter(dataset.shuffle(seed=args.seed, buffer_size=min(SHUFFLE_BUFFER_SIZE, count)))
        distinct_source = _distinct_documents(source, teacher, args.text_column)
        splits = {}
        for split_name, document_count in (("val", args.val_docs), ("test", args.test_docs)):
            splits[split_name] = write_split(
                split_name, document_count, distinct_source, args.text_column,
                output_dir, teacher, args.batch_size,
            )
            print(f"Cached {document_count} {split_name} documents")
        manifest = {
            "model": args.model,
            "model_revision": teacher.metadata["teacher_model_revision"],
            "transformers_version": transformers.__version__,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_revision": dataset_revision,
            "dataset_split": args.dataset_split,
            "data_files": args.data_files,
            "data_file_sha256": data_file_hashes,
            "text_column": args.text_column,
            "seed": args.seed,
            "max_length": teacher.max_length,
            "batch_size": args.batch_size,
            "shard_documents": SHARD_SIZE,
            "num_transformer_layers": teacher.num_layers,
            "hidden_width": teacher.hidden_width,
            "cached_layers": teacher.layers,
            "layer_indexing": teacher.metadata["layer_indexing"],
            "activation_shape": "[documents, cached_layers, hidden_width]",
            "token_ids_layout": "flat unpadded input_ids; input_offsets delimit each document",
            "teacher_parameter_dtypes": teacher.metadata["teacher_parameter_dtypes"],
            "saved_tensor_dtypes": next(info["tensor_dtypes"] for info in splits.values() if info["documents"]),
            "final_token": teacher.metadata["final_token"],
            "output_head_loaded_but_not_run": True,
            "split_identity": "disjoint SHA-256 of capped unpadded little-endian int64 token IDs",
            "teacher": teacher.metadata,
            "splits": splits,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Cache written to {output_dir}")
    finally:
        for item in (distinct_source, source):
            close = getattr(item, "close", None)
            if close is not None:
                close()
        teacher.close()
        del distinct_source, source, dataset
        gc.collect()


if __name__ == "__main__":
    main()
