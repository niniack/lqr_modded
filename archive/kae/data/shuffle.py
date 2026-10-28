"""HF-compatible bounded shuffle with an explicit, durable buffer."""

from copy import deepcopy

import numpy as np


class CheckpointableShuffle:
    """Keep HF's shard order and row shuffle, including its batched RNG draws.

    Datasets 4.4.2 only checkpoints the input cursor of its shuffle iterable. We
    retain its source/shard preparation but own the missing buffer and RNG state.
    Replacements happen before returning a row, so a snapshot never needs to
    resume inside a suspended generator's post-yield buffer update.
    """

    def __init__(self, dataset, *, seed, buffer_size, resume_state=None):
        self._dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        # Pinned HF wraps the row source in BufferShuffledExamplesIterable.
        # Remove only that buffer, retaining its Arrow rebatching and the
        # dataset's ShufflingConfig, which applies exactly the same shard order.
        self._dataset._ex_iterable = self._dataset._ex_iterable.ex_iterable
        self._buffer_size = buffer_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._buffer = []
        self._indices = []
        self._index_position = 0
        self._draining = False
        self._drain_position = 0
        self._source_started = False
        self._starting_source_state = None
        if resume_state is not None:
            if (resume_state["version"] != 1 or resume_state["seed"] != seed
                    or resume_state["buffer_size"] != buffer_size):
                raise ValueError("Prompt shuffle checkpoint does not match this source")
            self._rng.bit_generator.state = deepcopy(resume_state["rng"])
            self._buffer = deepcopy(resume_state["buffer"])
            self._indices = list(resume_state["indices"])
            self._index_position = resume_state["index_position"]
            self._draining = resume_state["draining"]
            self._drain_position = resume_state["drain_position"]
            self._starting_source_state = deepcopy(resume_state["source"])
            if self._starting_source_state is not None:
                self._dataset.load_state_dict(self._starting_source_state)
        self._source = iter(self._dataset)

    def __iter__(self):
        return self

    def _next_source(self):
        self._source_started = True
        return next(self._source)

    def __next__(self):
        if not self._draining:
            try:
                while len(self._buffer) < self._buffer_size:
                    self._buffer.append(self._next_source())
                incoming = self._next_source()
            except StopIteration:
                self._rng.shuffle(self._buffer)
                self._draining = True
            else:
                if self._index_position == len(self._indices):
                    # Match HF's _iter_random_indices(..., random_batch_size=1000),
                    # including unused draws when the source reaches its end.
                    self._indices = self._rng.integers(0, self._buffer_size, size=1000).tolist()
                    self._index_position = 0
                index = self._indices[self._index_position]
                self._index_position += 1
                outgoing = self._buffer[index]
                self._buffer[index] = incoming
                return outgoing
        if self._drain_position == len(self._buffer):
            raise StopIteration
        outgoing = self._buffer[self._drain_position]
        self._drain_position += 1
        return outgoing

    def state_dict(self) -> dict:
        source_state = (self._dataset.state_dict() if self._source_started
                        else self._starting_source_state)
        return {
            "version": 1,
            "seed": self._seed,
            "buffer_size": self._buffer_size,
            "source": deepcopy(source_state),
            "rng": deepcopy(self._rng.bit_generator.state),
            "buffer": deepcopy(self._buffer),
            "indices": self._indices.copy(),
            "index_position": self._index_position,
            "draining": self._draining,
            "drain_position": self._drain_position,
        }

    def close(self):
        close = getattr(self._source, "close", None)
        if close is not None:
            close()
        self._buffer.clear()
