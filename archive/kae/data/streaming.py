"""Dictionary targets from one persistent, training-only prompt stream."""

from collections import deque
from copy import deepcopy

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .datasets import DictionaryDataset
from .prompts import PromptStream, SHUFFLE_BUFFER_SIZE
from ..progress import TrainingProgress


def _cpu_batch(batch):
    return {key: value.detach().to(device="cpu", copy=True) for key, value in batch.items()}


class StreamingActivations(IterableDataset):
    """Logical epochs over one persistent source of frozen clean teacher states.

    The fixed CPU probe is the first accepted training documents. Its targets
    are reused by the first epoch, so no calibration or duplicate extraction is
    needed. Optional raw prefix context is shared with the same probe targets.
    """

    def __init__(
        self, cache, teacher, *, dataset, dataset_config=None,
        dataset_split="train", data_files=None, text_column="text",
        docs_per_epoch, seed, dataset_revision=None, teacher_batch_size=8,
        reporter=None, min_truncation=2, crop_seed=None, resume_state=None,
    ):
        super().__init__()
        self._require_main_process()
        for name, value in (("docs_per_epoch", docs_per_epoch),
                            ("teacher_batch_size", teacher_batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must name a Hugging Face dataset or builder")
        if not isinstance(text_column, str) or not text_column:
            raise ValueError("text_column must be non-empty")
        self.teacher = teacher
        self.docs_per_epoch = docs_per_epoch
        self.teacher_batch_size = teacher_batch_size
        self.reporter = reporter or TrainingProgress()
        self._pending = deque()
        self._prompts = None
        self._closed = False
        self._iterator_active = False
        self._documents_seen = self._tokens_seen = self._heldout_prompts_skipped = 0
        self._prefix_bin_width = max(1, (teacher.max_length + 31) // 32)
        self._prefix_edges = list(range(0, teacher.max_length, self._prefix_bin_width)) + [teacher.max_length]
        self._prefix_counts = [0] * (len(self._prefix_edges) - 1)
        self._probe_position = 0
        heldout = cache.heldout_hashes()
        probe_count = min(256, docs_per_epoch)
        self.run_config = {
            "dataset": dataset, "dataset_config": dataset_config,
            "dataset_split": dataset_split, "dataset_revision": dataset_revision,
            "data_files": data_files, "text_column": text_column,
            "source_data_sha256": None,
            "training_data_mode": "streaming",
            "epoch_definition": "Fixed accepted-document budget; one persistent shuffled source, never replayed",
            "num_train_docs": docs_per_epoch,
            "num_train_docs_definition": "Accepted documents per logical epoch, not corpus cardinality",
            "num_train_eval_docs": probe_count,
            "max_length": teacher.max_length,
            "streaming_shuffle_buffer": SHUFFLE_BUFFER_SIZE,
            "streaming_shuffle_seed": seed,
            "teacher_batch_size": teacher_batch_size,
            "heldout_prompt_hash_count": len(heldout),
            "token_hash_definition": "SHA-256 of capped unpadded little-endian int64 tokens before crop",
        }
        if resume_state is not None:
            if resume_state.get("version") != 2:
                raise ValueError(
                    "Clean streaming activations require checkpoint version 2; "
                    "old edit-bearing stream checkpoints cannot be resumed"
                )
            if any(
                resume_state["run_config"].get(key) != value for key, value in self.run_config.items()
            ):
                raise ValueError("Streaming activation checkpoint does not match the training recipe")
        try:
            self._prompts = PromptStream(
                teacher, heldout, self.reporter, dataset=dataset,
                dataset_config=dataset_config, dataset_split=dataset_split,
                data_files=data_files, dataset_revision=dataset_revision,
                text_column=text_column, seed=seed, docs_per_epoch=docs_per_epoch,
                min_truncation=min_truncation, crop_seed=seed if crop_seed is None else crop_seed,
                resume_state=None if resume_state is None else resume_state["prompts"],
            )
            self.run_config["crop_protocol"] = self._prompts.metadata
            if resume_state is not None:
                self._restore_state(resume_state)
                self.reporter.report(force=True, phase="train_probe_restored", probe_documents=probe_count)
                return
            batches, records = [], []
            self.reporter.report(force=True, phase="preparing_train_probe", probe_documents=probe_count)
            for start in range(0, probe_count, teacher_batch_size):
                batch, prompts = self._extract_batch(min(teacher_batch_size, probe_count - start))
                batches.append(batch)
                records.extend(prompts)
            self.probe_dataset = DictionaryDataset.from_batches(batches)
            self._probe_lengths = tuple(record[1] for record in records)
            self._probe_skipped = tuple(record[2] for record in records)
            self.run_config["train_probe_token_sha256"] = [record[3].hex() for record in records]
            self.run_config["train_probe_crop_lengths"] = list(self._probe_lengths)
            self.reporter.report(force=True, phase="train_probe_ready", probe_documents=probe_count)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _require_main_process():
        if get_worker_info() is not None:
            raise RuntimeError("StreamingActivations requires DataLoader(num_workers=0)")

    def __len__(self):
        return self.docs_per_epoch

    def state_dict(self) -> dict:
        """Snapshot the persistent source and all already-extracted CPU targets."""
        self._require_main_process()
        if self._closed:
            raise RuntimeError("Cannot checkpoint closed streaming activations")
        return {
            "version": 2,
            "prompts": self._prompts.state_dict(),
            "run_config": deepcopy(self.run_config),
            "probe_dataset": dict(self.probe_dataset.tensors),
            "probe_lengths": self._probe_lengths,
            "probe_skipped": self._probe_skipped,
            "probe_position": self._probe_position,
            "pending": list(self._pending),
            "documents_seen": self._documents_seen,
            "tokens_seen": self._tokens_seen,
            "heldout_prompts_skipped": self._heldout_prompts_skipped,
            "prefix_counts": self._prefix_counts.copy(),
        }

    def _restore_state(self, state):
        # Reuse immutable CPU targets; never regenerate probe or pending batches.
        self.probe_dataset = DictionaryDataset(dict(state["probe_dataset"]))
        self._probe_lengths = tuple(state["probe_lengths"])
        self._probe_skipped = tuple(state["probe_skipped"])
        self._probe_position = state["probe_position"]
        self._pending = deque(state["pending"])
        self._documents_seen = state["documents_seen"]
        self._tokens_seen = state["tokens_seen"]
        self._heldout_prompts_skipped = state["heldout_prompts_skipped"]
        self._prefix_counts = list(state["prefix_counts"])
        self.run_config = deepcopy(state["run_config"])

    def _extract_batch(self, count):
        prompts = [self._prompts.next_prompt() for _ in range(count)]
        tokens = [prompt[0] for prompt in prompts]
        batch = self.reporter.call(
            "teacher_targets", lambda: _cpu_batch(self.teacher.targets_batch(tokens))
        )
        self._prompts._accepted_from_source += count
        self.reporter.count(teacher_documents=count,
                            teacher_tokens=sum(prompt[1] for prompt in prompts), teacher_batches=1)
        self.reporter.report(last_teacher_batch_size=count)
        return batch, prompts

    def __iter__(self):
        self._require_main_process()
        if self._closed:
            raise RuntimeError("StreamingActivations is closed")
        if self._iterator_active:
            raise RuntimeError("StreamingActivations supports only one active training iterator")
        self._iterator_active = True
        try:
            for position in range(self.docs_per_epoch):
                if self._probe_position < len(self.probe_dataset):
                    index = self._probe_position
                    targets = self.probe_dataset[index]
                    length, skipped = self._probe_lengths[index], self._probe_skipped[index]
                    self._probe_position += 1
                else:
                    if not self._pending:
                        batch, prompts = self._extract_batch(
                            min(self.teacher_batch_size, self.docs_per_epoch - position)
                        )
                        self._pending.extend(
                            ({key: value[index] for key, value in batch.items()}, prompt[1], prompt[2])
                            for index, prompt in enumerate(prompts)
                        )
                    targets, length, skipped = self._pending.popleft()
                self._documents_seen += 1
                self._tokens_seen += length
                self._heldout_prompts_skipped += skipped
                bucket = min(length // self._prefix_bin_width, len(self._prefix_counts) - 1)
                self._prefix_counts[bucket] += 1
                yield targets
        except Exception:
            self.close()
            raise
        finally:
            self._iterator_active = False

    def progress(self):
        return {
            "train/documents_seen": self._documents_seen,
            "train/tokens_seen": self._tokens_seen,
            "train/heldout_prompts_skipped": self._heldout_prompts_skipped,
            "train/prefix_length_histogram": {
                "counts": self._prefix_counts.copy(), "edges": self._prefix_edges,
            },
        }

    def close(self):
        self._closed = True
        self._pending.clear()
        if self._prompts is not None:
            self._prompts.close()
