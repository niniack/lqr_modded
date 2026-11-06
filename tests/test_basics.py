from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from extensions.data import LayerInputs
from extensions.steering import (
    DriftLQRSteering, QwenLQRSteering, completion_metrics, riccati, trace,
)


@pytest.fixture
def setup():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, pad_token_id=0,
                             eos_token_id=2)
        model = Qwen3ForCausalLM(config)
        model = model.eval().requires_grad_(False)
        nominal = deepcopy(model)
        E = torch.randn(4, 32) * 0.02
        A = torch.eye(32).expand(3, -1, -1)
        inputs = {'input_ids': torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])}
        yield model, nominal, E, A, inputs
    torch.set_num_threads(threads)


def test_zero_drift_weight_reproduces_plain_tokens(setup):
    model, nominal, E, A, inputs = setup
    base = QwenLQRSteering(model, A, E, 0.1, 10, 0.1)
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
    assert len(controller.drift_log) == len(A) * 8
    for _, drift, orthogonal in controller.drift_log:
        torch.testing.assert_close(drift, torch.zeros_like(drift), rtol=0, atol=1e-7)
        torch.testing.assert_close(orthogonal, torch.zeros_like(orthogonal), rtol=0, atol=1e-7)


@pytest.mark.parametrize("last_n", [1, 2, 3])
@pytest.mark.parametrize("every", [1, 2])
def test_last_layers_preserve_prefix_and_zero_weight_tokens(setup, last_n, every):
    model, nominal, E, A, inputs = setup
    base = QwenLQRSteering(model, A, E, 0.1, 10, 0.1,
                           steer_last_n_layers=last_n, every=every)
    drift = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1,
                            steer_last_n_layers=last_n, every=every, drift_weight=0)
    with LayerInputs(model) as record:
        model.base_model(**inputs, use_cache=False)
    clean = record.states
    for controller in (base, drift):
        with LayerInputs(model) as record, controller.active(4):
            model.base_model(**inputs, use_cache=False)
        steered = record.states
        first = len(model.model.layers) - last_n
        for layer in range(first + 1):
            assert torch.equal(clean[layer], steered[layer])
        assert not torch.equal(clean[-1], steered[-1])
        assert list(controller.steered) == list(range(first, len(A), every))
        assert set(controller.K) == set(controller.steered)
    outputs = []
    for controller in (base, drift):
        torch.manual_seed(9)
        with controller.active(4):
            outputs.append(model.generate(**inputs, max_new_tokens=8, do_sample=True,
                                          eos_token_id=None, pad_token_id=0))
    assert torch.equal(*outputs)


def _frozen_gains(A, q, r, qf, actuated):
    """Frozen recurrence, checking the gains actually consumed by controllers."""
    eye = torch.eye(A.shape[-1], dtype=torch.float64)
    S = qf * eye
    gains = {}
    for layer in reversed(range(len(A))):
        a = A[layer].double()
        F = S @ a
        K = (torch.linalg.solve(S + r * eye, F)
             if layer in actuated else torch.zeros_like(a))
        gains[layer] = K
        S = q * eye + a.T @ F - F.T @ K
        S = (S + S.T) / 2
    return gains


@pytest.mark.parametrize("every,last_n", [(1, None), (1, 2), (2, 3), (2, 2)])
def test_streamed_gains_preserve_sparse_planning(setup, every, last_n):
    model, nominal, E, A, _ = setup
    # Non-diagonal dynamics expose planning through unactuated blocks.
    A = A + torch.randn_like(A) * 0.04
    controller = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.2,
                                 every=every, steer_last_n_layers=last_n)
    first = 0 if last_n is None else len(A) - last_n
    actuated = range(first, len(A), every)
    assert list(controller.steered) == list(actuated)
    expected = _frozen_gains(A, 0.1, 10, 0.2, actuated)
    actual = list(riccati(A, 0.1, 10, 0.2, actuated=actuated))
    assert [layer for layer, _ in actual] == list(reversed(range(len(A))))
    for layer, K in actual:
        assert K.dtype == torch.float64
        torch.testing.assert_close(K, expected[layer], rtol=1e-12, atol=1e-14)
        if layer in actuated:
            torch.testing.assert_close(controller.K[layer], expected[layer].float())
        else:
            assert layer not in controller.K
    if every == 2:
        fully_actuated = _frozen_gains(A, 0.1, 10, 0.2, range(len(A)))
        assert not torch.allclose(expected[first], fully_actuated[first])


def test_drift_push_uses_only_orthogonal_displacement(setup):
    model, nominal, E, A, _ = setup
    controller = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1, drift_weight=2)
    controller.lmbda = 4
    for layer in controller.steered:
        x_nom = torch.randn(2, 1, E.shape[-1])
        x = x_nom + torch.randn_like(x_nom) * 0.1
        controller._x_nom[layer] = x_nom
        v = controller.E_unit[layer]
        d = x - x_nom
        off = d - (d @ v)[..., None] * v
        plain = QwenLQRSteering.push(controller, layer, x)
        expected = plain - 2 * (off @ controller.K[layer].T)
        torch.testing.assert_close(controller.push(layer, x), expected)


def test_dry_run_trace_preserves_eos_and_metrics(setup):
    model, nominal, E, A, inputs = setup
    controller = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1, drift_weight=2)
    controller.dry_run = True
    generated = torch.tensor([[12, 2, 0], [13, 14, 15]])
    sequences = torch.cat([inputs["input_ids"], generated], dim=1)
    replay = trace(model, controller, 4, sequences, inputs["input_ids"].shape[1])
    assert torch.equal(replay["valid"], torch.tensor([[True, True, False], [True, True, True]]))
    torch.testing.assert_close(replay["steered"], replay["nominal"], rtol=0, atol=0)
    log_length = len(controller.drift_log)
    assert log_length == len(A)
    metrics = completion_metrics(replay)
    assert len(metrics) == len(sequences)
    for row in metrics:
        assert row == dict(along_v=0, change=0, orthogonal=0, push_energy=0)
    assert len(controller.drift_log) == log_length


def test_nominal_must_be_an_independent_model(setup):
    model, _, E, A, _ = setup
    with pytest.raises(ValueError, match="separate"):
        DriftLQRSteering(model, model, A, E, 0.1, 10, 0.1)


def test_base_push_tracks_linear_setpoint_with_full_gain(setup):
    model, _, E, A, _ = setup
    A = A + torch.randn_like(A) * 0.04
    controller = QwenLQRSteering(model, A, E, 0.1, 10, 0.2, every=2)
    controller.lmbda = 4
    x = torch.randn(2, 3, E.shape[-1])
    for layer in range(len(A)):
        actual = controller.push(layer, x)
        assert actual.dtype == torch.float32
        if layer not in controller.steered:
            assert torch.count_nonzero(actual) == 0
            continue
        v = controller.E_unit[layer]
        error = (4 * E[layer].norm() - x @ v)[..., None] * v
        expected = error @ controller.K[layer].T
        torch.testing.assert_close(actual, expected)
        target = (4 * E[layer]).expand_as(x)
        torch.testing.assert_close(controller.push(layer, target), torch.zeros_like(x),
                                   rtol=0, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_generation_matches_full_sequence_replay(setup, dtype):
    model, nominal, E, A, inputs = setup
    model.to(dtype=dtype)
    nominal.to(dtype=dtype)
    controller = DriftLQRSteering(model, nominal, A, E, 0.1, 10, 0.1,
                                 drift_weight=2, every=2, steer_last_n_layers=2)
    with LayerInputs(model) as record, controller.active(4):
        sequences = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                   use_cache=True, eos_token_id=None, pad_token_id=0)
    cached = torch.stack([state[:, -1] for state in record.states])
    assert cached.dtype == dtype
    replay = trace(model, controller, 4, sequences, inputs["input_ids"].shape[1])
    tolerance = dict(rtol=1e-5, atol=1e-7) if dtype == torch.float32 else dict(rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(replay["steered"][:, :, -1], cached, **tolerance)


def test_controller_context_removes_steering_after_exception(setup):
    model, _, E, A, inputs = setup
    controller = QwenLQRSteering(model, A, E, 0.1, 10, 0.1)
    clean = model(**inputs, use_cache=False).logits
    with pytest.raises(RuntimeError, match="interrupted"):
        with controller.active(4):
            steered = model(**inputs, use_cache=False).logits
            assert not torch.equal(steered, clean)
            raise RuntimeError("interrupted")
    assert torch.equal(model(**inputs, use_cache=False).logits, clean)
    with controller.active(4):
        assert torch.equal(model(**inputs, use_cache=False).logits, steered)
