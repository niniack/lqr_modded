"""KL training must update the student through the frozen teacher head."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kae.models import AIKAE, KoopmanOperator
from kae.train import train_one_epoch
from kae.teacher import TeacherHead


class KLTrainingTest(unittest.TestCase):
    def test_kl_only_updates_student_through_frozen_teacher_head(self):
        model = AIKAE(hidden_width=2, latent_width=2, operator=KoopmanOperator(2, steps=2))
        lm_head = nn.Linear(2, 2, bias=False)
        nn.init.eye_(lm_head.weight)
        teacher = TeacherHead(nn.Identity(), lm_head, None).eval().requires_grad_(False)
        head_grad_modes = []
        teacher.register_forward_hook(
            lambda module, args, output: head_grad_modes.append(output.requires_grad)
        )
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = inputs.flip(-1).unsqueeze(1).expand(-1, 2, -1).clone()
        student_before = model.rollout(inputs)[:, -1].detach()
        log_p = torch.log_softmax(targets[:, -1], dim=-1)
        log_q = torch.log_softmax(student_before, dim=-1)
        expected_kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().item()
        transition_before = model.operator.transition.weight.detach().clone()
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        metrics = train_one_epoch(
            model,
            loader,
            torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            output_layers=[15, 16],
            state_weight=0,
            jvp_weight=0,
            kl_weight=2.5,
            teacher=teacher,
            record_diagnostics=True,
        )

        self.assertGreater(expected_kl, 0)
        self.assertAlmostEqual(metrics["train/kl_divergence"], expected_kl)
        self.assertAlmostEqual(metrics["train/loss"], 2.5 * expected_kl)
        self.assertGreater(metrics["train/state_rollout_mse"], 0)
        self.assertFalse(torch.allclose(model.operator.transition.weight, transition_before))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertEqual(head_grad_modes, [True, False])

    def test_logged_only_kl_does_not_build_a_gradient_graph(self):
        model = AIKAE(hidden_width=2, latent_width=2, operator=KoopmanOperator(2, steps=2))
        lm_head = nn.Linear(2, 2, bias=False)
        nn.init.eye_(lm_head.weight)
        teacher = TeacherHead(nn.Identity(), lm_head, None).eval().requires_grad_(False)
        head_grad_modes = []
        teacher.register_forward_hook(
            lambda module, args, output: head_grad_modes.append(output.requires_grad)
        )
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = inputs.flip(-1).unsqueeze(1).expand(-1, 2, -1).clone()
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        metrics = train_one_epoch(
            model,
            loader,
            torch.optim.SGD(model.parameters(), lr=0),
            device="cpu",
            output_layers=[15, 16],
            state_weight=1,
            kl_weight=0,
            teacher=teacher,
            record_diagnostics=True,
        )

        self.assertEqual(head_grad_modes, [False, False])
        self.assertGreater(metrics["train/kl_divergence"], 0)
        self.assertAlmostEqual(metrics["train/loss"], metrics["train/state_rollout_mse"])


if __name__ == "__main__":
    unittest.main()
