from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm.linear_prediction import capture, jvp, reference
from llm.model import blocks
from llm.steering import DriftLQRSteering, LQRSteering


@pytest.fixture
def setup():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, pad_token_id=0)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        nominal = deepcopy(model)
        E = torch.randn(4, 32) * 0.02
        A = torch.eye(32).expand(3, -1, -1)
        inputs = {'input_ids': torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])}
        yield model, nominal, E, A, inputs
    torch.set_num_threads(threads)


def test_zero_drift_weight_reproduces_plain_tokens(setup):
    model, nominal, E, A, inputs = setup
    base = LQRSteering(model, A, E, 0.1, 10, 0.1)
    drift = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1, drift_weight=0)
    outputs = []
    for controller in (base, drift):
        torch.manual_seed(9)
        with controller.active(4):
            outputs.append(model.generate(**inputs, max_new_tokens=8, do_sample=True,
                                          eos_token_id=None, pad_token_id=0))
    assert torch.equal(*outputs)


def test_nominal_copy_dry_run_has_zero_drift(setup):
    model, nominal, E, A, inputs = setup
    controller = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1, drift_weight=2)
    controller.dry_run = True
    with controller.active(4):
        model.generate(**inputs, max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)
    for _, drift, orthogonal in controller.drift_log:
        torch.testing.assert_close(drift, torch.zeros_like(drift), rtol=0, atol=1e-7)
        torch.testing.assert_close(orthogonal, torch.zeros_like(orthogonal), rtol=0, atol=1e-7)


def test_zero_replay_is_nominal_and_unit_replay_matches_controller(setup):
    model, _, E, A, inputs = setup
    controller = LQRSteering(model, A, E, 0.1, 10, 0.1)
    nominal, _ = capture(model, inputs)
    controlled, pushes = reference(model, inputs, controller)
    zero, _ = capture(model, inputs, pushes * 0)
    unit, _ = capture(model, inputs, pushes)
    for expected, replayed in zip(nominal, zero):
        assert torch.equal(expected, replayed)
    for expected, replayed in zip(controlled, unit):
        torch.testing.assert_close(expected, replayed, rtol=1e-6, atol=1e-7)


def test_prompt_jacobian_one_step_error_is_small(setup):
    model, _, E, A, inputs = setup
    controller = LQRSteering(model, A, E, 0.1, 10, 0.1)
    nominal, contexts = capture(model, inputs, linearize=True)
    _, pushes = reference(model, inputs, controller)
    states, _ = capture(model, inputs, pushes * 0.01)
    d = [x[:, -1] - clean[:, -1] for x, clean in zip(states, nominal)]
    for layer, block in enumerate(blocks(model)):
        prediction = jvp(block, contexts[layer], d[layer], layer) + pushes[layer] * 0.01
        error = (prediction.double() - d[layer + 1].double()).norm(dim=-1) / d[layer + 1].double().norm(dim=-1)
        assert error.median().item() < 0.05, (layer, error)


@pytest.mark.parametrize("last_n", [1, 2, 3])
def test_last_layers_preserve_prefix_and_zero_weight_tokens(setup, last_n):
    model, nominal, E, A, inputs = setup
    base = LQRSteering(model, A, E, 0.1, 10, 0.1, steer_last_n_layers=last_n)
    drift = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1,
                            steer_last_n_layers=last_n, drift_weight=0)
    clean, _ = capture(model, inputs)
    for controller in (base, drift):
        steered, pushes = reference(model, inputs, controller)
        first = len(blocks(model)) - last_n
        for layer in range(first + 1):
            assert torch.equal(clean[layer], steered[layer])
        assert not torch.equal(clean[-1], steered[-1])
        assert torch.count_nonzero(pushes[:first]) == 0
    outputs = []
    for controller in (base, drift):
        torch.manual_seed(9)
        with controller.active(4):
            outputs.append(model.generate(**inputs, max_new_tokens=8, do_sample=True,
                                          eos_token_id=None, pad_token_id=0))
    assert torch.equal(*outputs)
