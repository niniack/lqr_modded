"""Numerical checks for reconstruction, rollout, and gradient diagnostics."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from kae.models import AIKAE, KoopmanOperator
from kae.train import train_one_epoch


class ShiftedDecoder(AIKAE):
    """Make reconstruction imperfect while preserving deterministic rollouts."""

    def __init__(self):
        super().__init__(hidden_width=2, latent_width=2, operator=KoopmanOperator(2, steps=2))
        self.register_buffer("decode_offset", torch.tensor([1.0, 0.0]))

    def decode(self, z):
        return super().decode(z) + self.decode_offset


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_fvu_uses_epoch_wide_per_feature_target_means(self):
        model = AIKAE(hidden_width=2, latent_width=2, operator=KoopmanOperator(2, steps=2))
        inputs = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        targets = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]]
        )
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)

        metrics = train_one_epoch(
            model,
            loader,
            torch.optim.SGD(model.parameters(), lr=0),
            device="cpu",
            output_layers=[15, 16],
            record_diagnostics=True,
        )

        # One document per batch has zero within-batch variation; the epoch does not.
        self.assertAlmostEqual(metrics["train/state_mse_first"], 0.5)
        self.assertAlmostEqual(metrics["train/state_fvu_first"], 1)
        self.assertAlmostEqual(metrics["train/state_mse/layer_15"], 0.5)
        self.assertAlmostEqual(metrics["train/state_mse/layer_16"], 2)
        self.assertAlmostEqual(metrics["train/state_fvu/layer_15"], 1)
        self.assertAlmostEqual(metrics["train/state_fvu/layer_16"], 4)
        self.assertAlmostEqual(metrics["train/state_rollout_mse"], 1.25)
        self.assertAlmostEqual(metrics["train/state_rollout_fvu"], 2.5)
        self.assertAlmostEqual(metrics["train/latent_mse"], 1.25)
        self.assertAlmostEqual(metrics["train/latent_fvu"], 2.5)
        self.assertAlmostEqual(metrics["train/reconstruction_mse"], 0)
        self.assertAlmostEqual(metrics["train/reconstruction_fvu"], 0)
        self.assertGreater(metrics["train/grad_norm/operator"], 0)
        self.assertGreaterEqual(
            metrics["train/grad_norm/total"], metrics["train/grad_norm/operator"]
        )


    def test_reconstruction_is_logged_against_inputs_but_not_optimized(self):
        model = ShiftedDecoder()
        inputs = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        shifted = inputs + model.decode_offset
        targets = shifted.unsqueeze(1).expand(-1, 2, -1).clone()
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)

        metrics = train_one_epoch(
            model,
            loader,
            torch.optim.SGD(model.parameters(), lr=0),
            device="cpu",
            output_layers=[15, 16],
            record_diagnostics=True,
        )

        self.assertAlmostEqual(metrics["train/loss"], 0)
        self.assertAlmostEqual(metrics["train/state_rollout_mse"], 0)
        self.assertAlmostEqual(metrics["train/reconstruction_mse"], 0.5)
        self.assertAlmostEqual(metrics["train/reconstruction_fvu"], 1)


if __name__ == "__main__":
    unittest.main()
