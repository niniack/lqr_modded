"""Stable identities for unpadded token sequences and held-out exclusion."""

import hashlib
import torch


def token_hash(token_ids: torch.Tensor) -> bytes:
    """Hash the complete unpadded integer sequence, independent of tensor layout."""
    ids = token_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
    return hashlib.sha256(ids.numpy().astype("<i8", copy=False).tobytes()).digest()


