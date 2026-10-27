"""Tensor dictionaries with ordinary DataLoader collation."""

import torch
from torch.utils.data import Dataset


class DictionaryDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        if not tensors:
            raise ValueError("A tensor dictionary must not be empty")
        lengths = {len(tensor) for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("All dataset tensors must have the same document count")
        self.tensors = tensors
        self._length = lengths.pop()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    @classmethod
    def from_batches(cls, batches):
        if not batches:
            raise ValueError("Cannot build a dataset without batches")
        keys = batches[0].keys()
        if any(batch.keys() != keys for batch in batches):
            raise ValueError("Batches must have identical dictionary keys")
        return cls({key: torch.cat([batch[key] for batch in batches]) for key in keys})


def attach_context(dataset, teacher, records, *, batch_size=8, reporter=None):
    """Enrich fixed targets using their exact stored tokens, without replacing targets.

    ``records`` follows dataset order and contains unpadded CPU ``input_ids [T]``.
    Already-enriched datasets are returned unchanged; new context tensors live on
    CPU and all original tensors are shared with the returned dictionary dataset.
    """
    if len(records) != len(dataset):
        raise ValueError("Context records must match the dataset count")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Context batch_size must be a positive integer")
    context_keys = {"context", "context_mask"}
    present = context_keys.intersection(dataset.tensors)
    if present:
        if present != context_keys:
            raise ValueError("Context datasets must contain both context and context_mask")
        return dataset
    if not records:
        raise ValueError("Cannot attach context to an empty dataset")
    tokens = []
    for record in records:
        ids = record["input_ids"]
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.device.type != "cpu":
            raise ValueError("Context records require unpadded one-dimensional CPU input_ids")
        if not 1 <= ids.numel() <= teacher.max_length:
            raise ValueError("Context tokens must be nonempty and within teacher max_length")
        ids = ids.unsqueeze(0)
        tokens.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    tensors = None
    for start in range(0, len(tokens), batch_size):
        subset = tokens[start:start + batch_size]
        batch = (teacher.contexts_batch(subset) if reporter is None else
                 reporter.call("context_collection", teacher.contexts_batch, subset))
        if tensors is None:
            tensors = {
                key: value.new_empty((len(dataset), *value.shape[1:]), device="cpu")
                for key, value in batch.items()
            }
        for key, value in batch.items():
            tensors[key][start:start + len(subset)].copy_(value.detach())
        if reporter is not None:
            reporter.report(context_documents=start + len(subset), context_document_budget=len(dataset))
    return DictionaryDataset({**dataset.tensors, **tensors})
