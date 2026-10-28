"""One persistent bounded prompt source, excluding held-out IDs before inference."""

import hashlib
import itertools
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset_builder
from datasets.builder import Key
from datasets.packaged_modules.parquet.parquet import Parquet

from .shuffle import CheckpointableShuffle
from .tokens import token_hash

SHUFFLE_BUFFER_SIZE = 10_000


class _SynchronousParquet(Parquet):
    """Keep HF split/schema handling without Scanner background file callbacks."""

    def _generate_tables(self, files):
        for file_index, file in enumerate(itertools.chain.from_iterable(files)):
            # DatasetBuilder supplies its normal authenticated streaming `open`.
            with open(file, "rb") as source:
                reader = pq.ParquetFile(source, pre_buffer=False)
                try:
                    if reader.metadata.num_row_groups:
                        batch_size = self.config.batch_size or reader.metadata.row_group(0).num_rows
                        for batch_index, batch in enumerate(
                            reader.iter_batches(
                                batch_size=batch_size,
                                columns=self.config.columns,
                                use_threads=False,
                            )
                        ):
                            yield Key(file_index, batch_index), self._cast_table(pa.Table.from_batches([batch]))
                finally:
                    reader.close()


def load_prompt_stream(path, name=None, *, split, **kwargs):
    """Resolve sources with HF; read unfiltered Parquet synchronously.

    Non-Parquet builders retain their native streaming reader. Parquet filters,
    Scanner options and file-skipping policies are not supported by this reader:
    rejecting them prevents silently changing which documents enter training.
    """
    builder = load_dataset_builder(path, name, **kwargs)
    if isinstance(builder, Parquet):
        config = builder.config
        if (
            config.filters is not None
            or config.fragment_scan_options is not None
            or config.on_bad_files != "error"
        ):
            raise ValueError(
                "Synchronous Parquet streaming requires no row filters or Scanner options "
                "and on_bad_files='error'"
            )
        config_kwargs = {
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.name != "name"
        }
        builder = _SynchronousParquet(
            cache_dir=kwargs.get("cache_dir"),
            dataset_name=builder.dataset_name,
            config_name=config.name,
            hash=builder.hash,
            config_id=builder.config_id,
            base_path=builder.base_path,
            info=builder.info,
            token=builder.token,
            repo_id=builder.repo_id,
            storage_options=builder.storage_options,
            **config_kwargs,
        )
    return builder.as_streaming_dataset(split=split)


class PromptStream:
    def __init__(self, teacher, heldout, reporter, *, dataset, dataset_config,
                 dataset_split, data_files, dataset_revision, text_column,
                 seed, docs_per_epoch, min_truncation=2, crop_seed=None,
                 resume_state=None):
        self.teacher = teacher
        self._heldout = heldout
        self.reporter = reporter
        self._dataset_name = dataset
        self._dataset_split = dataset_split
        self._text_column = text_column
        self.docs_per_epoch = docs_per_epoch
        if type(min_truncation) is not int or min_truncation < 1:
            raise ValueError("min_truncation must be a positive integer")
        self.min_truncation = min_truncation
        self.crop_seed = crop_seed
        self.metadata = {
            "max_length": teacher.max_length,
            "min_truncation": min_truncation,
            "crop_seed": crop_seed,
            "crop_distribution": "uniform integer prefix length [min(min_truncation,T),T]",
            "heldout_identity": "SHA-256 of capped unpadded little-endian int64 tokens before crop",
            "crop_rng": "SHA-256(crop_seed:pre_crop_token_hash), private CPU torch.Generator",
        }
        self._source_config = {
            "dataset": dataset, "dataset_config": dataset_config,
            "dataset_split": dataset_split, "data_files": data_files,
            "dataset_revision": dataset_revision, "text_column": text_column,
            "seed": seed, "docs_per_epoch": docs_per_epoch,
            "crop_protocol": self.metadata,
        }
        if resume_state is not None and (
            resume_state["version"] != 1 or resume_state["source_config"] != self._source_config
        ):
            raise ValueError("Prompt stream checkpoint does not match this source and crop protocol")
        self._records_read = 0
        self._accepted_from_source = 0
        self._source = None
        self._dataset = None
        self.reporter.report(force=True, phase="opening_stream", dataset=dataset)
        self._dataset = self.reporter.call(
            "stream_open", load_prompt_stream, dataset, dataset_config,
            split=dataset_split, data_files=data_files, revision=dataset_revision,
        )
        self._source = CheckpointableShuffle(
            self._dataset, seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE,
            resume_state=None if resume_state is None else resume_state["shuffle"],
        )
        if resume_state is not None:
            self._records_read = resume_state["records_read"]
            self._accepted_from_source = resume_state["accepted_from_source"]

    def state_dict(self) -> dict:
        if self._source is None:
            raise RuntimeError("Cannot checkpoint a closed prompt stream")
        return {
            "version": 1,
            "source_config": deepcopy(self._source_config),
            "records_read": self._records_read,
            "accepted_from_source": self._accepted_from_source,
            "shuffle": self._source.state_dict(),
        }

    def close(self):
        source = self._source
        self._source = self._dataset = None
        close = getattr(source, "close", None)
        if close is not None:
            close()

    def next_prompt(self):
        skipped = 0
        while True:
            try:
                example = self.reporter.call("source_read", next, self._source)
            except StopIteration as error:
                raise ValueError(
                    f"Streaming dataset {self._dataset_name!r} split {self._dataset_split!r} "
                    f"exhausted after {self._records_read} source records and "
                    f"{self._accepted_from_source} accepted documents; "
                    f"cannot fill the fixed {self.docs_per_epoch}-document logical epoch "
                    "(including its fixed training probe). The source is never restarted."
                ) from error
            self._records_read += 1
            self.reporter.count(source_records=1)
            text = example.get(self._text_column) if isinstance(example, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Streaming dataset field {self._text_column!r} must contain non-empty text "
                    f"at source record {self._records_read} in split {self._dataset_split!r}"
                )
            tokenized = self.reporter.call("tokenize", self.teacher.tokenize, text)
            ids = tokenized.get("input_ids")
            mask = tokenized.get("attention_mask")
            if (
                not isinstance(ids, torch.Tensor)
                or not isinstance(mask, torch.Tensor)
                or ids.device.type != "cpu"
                or mask.device.type != "cpu"
                or ids.ndim != 2
                or ids.shape[0] != 1
                or ids.shape != mask.shape
            ):
                raise ValueError("GemmaTeacher.tokenize must return CPU input_ids/attention_mask of shape [1, T]")
            unpadded = ids[0][mask[0].bool()]
            length = unpadded.numel()
            if length == 0:
                raise ValueError(f"Streaming source record {self._records_read} tokenized to no valid tokens")
            prompt_hash = token_hash(unpadded)
            if prompt_hash in self._heldout:
                skipped += 1
                self.reporter.count(source_heldout_skips=1)
                continue
            # Held-out exclusion uses the complete capped identity, never the crop.
            if self.crop_seed is not None:
                digest = hashlib.sha256(
                    f"{self.crop_seed}:{prompt_hash.hex()}".encode()
                ).digest()
                generator = torch.Generator(device="cpu").manual_seed(
                    int.from_bytes(digest[:8], "little") % (2**63)
                )
                length = int(torch.randint(
                    min(self.min_truncation, length), length + 1, (), generator=generator
                ))
            tokenized = {
                "input_ids": unpadded[:length].unsqueeze(0),
                "attention_mask": torch.ones((1, length), dtype=mask.dtype),
            }
            return tokenized, length, skipped, prompt_hash

