"""Loss functions for KAE training."""

import torch
import torch.nn.functional as F


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise mean-squared error between matching tensors."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match; got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    return F.mse_loss(prediction, target)


def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(teacher || student), summing over the vocabulary dimension."""
    if student_logits.ndim < 1 or student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have matching [..., vocab_size] shapes; "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(
        student_log_probs, teacher_log_probs, reduction="none", log_target=True
    ).sum(dim=-1).mean()
