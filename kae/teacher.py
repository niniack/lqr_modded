"""Frozen teacher output head for raw cached final-token states."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class TeacherHead(nn.Module):
    """Apply the teacher's final norm, LM head, and optional logit softcap."""

    def __init__(self, norm: nn.Module, lm_head: nn.Module, softcap: float | None):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        self.softcap = softcap

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.norm(raw_state))
        if self.softcap is None:
            return logits
        return torch.tanh(logits / self.softcap) * self.softcap


def load_teacher_head(
    model_name: str, revision: str | None, device: torch.device | str
) -> TeacherHead:
    """Load a checkpoint on CPU; move only its frozen norm and head to device."""
    kwargs = {"dtype": torch.float32}
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    head = TeacherHead(
        model.model.norm,
        model.lm_head,
        getattr(model.config, "final_logit_softcapping", None),
    )
    del model
    return head.to(device).eval().requires_grad_(False)
