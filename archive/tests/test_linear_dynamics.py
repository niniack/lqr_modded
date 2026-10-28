import unittest

import torch

from scripts.fit_linear_dynamics import fit_block_dynamics


class LinearDynamicsTests(unittest.TestCase):
    def test_fitted_blocks_predict_unseen_column_states(self):
        generator = torch.Generator().manual_seed(17)
        expected = torch.randn(2, 4, 4, generator=generator) + torch.eye(4)
        states = [torch.randn(64, 4, generator=generator) + 2]
        for matrix in expected:
            states.append(states[-1] @ matrix.T)
        fitted, _ = fit_block_dynamics(torch.stack(states, dim=1))

        state = torch.randn(16, 4, generator=generator) - 3
        for actual, target in zip(fitted, expected):
            torch.testing.assert_close(state @ actual.T, state @ target.T, rtol=1e-4, atol=1e-4)
            state = state @ target.T

    def test_rank_deficient_fit_predicts_data_span_and_zero_on_unobserved_directions(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(32, 4, generator=generator)
        x[:, 2] = 0
        x[:, 3] = 2 * x[:, 0]
        target = torch.randn(4, 4, generator=generator)
        y = x @ target.T
        fitted, _ = fit_block_dynamics(torch.stack([x, y], dim=1))

        held_out = torch.randn(12, 4, generator=generator)
        held_out[:, 2] = 0
        held_out[:, 3] = 2 * held_out[:, 0]
        torch.testing.assert_close(held_out @ fitted[0].T, held_out @ target.T, rtol=1e-4, atol=1e-5)
        null_directions = torch.tensor([[-2.0, 0, 0, 1], [0, 0, 1.0, 0]])
        torch.testing.assert_close(null_directions @ fitted[0].T, torch.zeros(2, 4), rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
