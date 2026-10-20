"""Numerical checks for JVP training metrics."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kae.train import train_one_epoch


class LinearRollout(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def rollout(self, inputs):
        return torch.stack((self.scale * inputs, 2 * self.scale * inputs), dim=1)


class JvpTrainingMetricsTest(unittest.TestCase):
    def test_relative_error_uses_layerwise_sums_not_average_ratios(self):
        model = LinearRollout()
        inputs = torch.ones(2, 1)
        teacher_states = torch.zeros(2, 2, 1)
        directions = torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]])
        teacher_jvps = torch.tensor(
            [
                [[[1.0], [2.0]], [[2.0], [4.0]]],
                [[[10.0], [20.0]], [[20.0], [40.0]]],
            ]
        )
        loader = DataLoader(
            TensorDataset(inputs, teacher_states, directions, teacher_jvps),
            batch_size=1,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0)

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device="cpu",
            output_layers=[15, 16],
            state_weight=0,
            jvp_weight=1,
        )

        self.assertAlmostEqual(metrics["train/state_mse_first"], 4)
        self.assertAlmostEqual(metrics["train/state_mse/layer_15"], 4)
        self.assertAlmostEqual(metrics["train/state_mse/layer_16"], 16)
        self.assertAlmostEqual(metrics["train/state_rollout_mse"], 10)
        self.assertNotIn("train/state_mse", metrics)
        self.assertAlmostEqual(metrics["train/jvp_mse/layer_15"], 325 / 4)
        self.assertAlmostEqual(metrics["train/jvp_mse/layer_16"], 1300 / 4)
        self.assertAlmostEqual(metrics["train/jvp_relative_error/layer_15"], 325 / 505)
        self.assertAlmostEqual(metrics["train/jvp_relative_error/layer_16"], 1300 / 2020)
        self.assertAlmostEqual(metrics["train/jvp_relative_error"], 1625 / 2525)
        self.assertAlmostEqual(metrics["train/jvp_mse"], 1625 / 8)
        self.assertAlmostEqual(metrics["train/loss"], 1625 / 8)
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(torch.isfinite(model.scale.grad))


if __name__ == "__main__":
    unittest.main()
