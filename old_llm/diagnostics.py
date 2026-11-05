"""Batch-local, observational diagnostics for cached last-token steering.

States use raw decoder inputs and the final *pre-normalization* output. Controls
are measured after quantization, not read from a controller's requested push.
The nominal trajectory replays the exact consumed batch and forward schedule
with a fresh cache, never a teacher-forced full-sequence approximation. Optional
raw identity/mean-J/DMD/paired-DMD/steering-pair-DMD errors include changed KV
context and nonlinear effects; they are pooled residual RMS ratios, not isolated
operator accuracy scores.

An optional pair_sink (llm.steering_dmd.PairSink) receives each layer's
(delta, push, delta_next) tensors during summarize(), for steering-pair DMD.

Effort is the token mean of sum_l ||u_l||^2 / ||x_l||^2 in raw coordinates.
peak_push_layer identifies the largest relative push across valid tokens and
layers, zero-based; ties choose the lowest layer, and no push gives None.
Group summaries use the most common defined peak layer, never its mean.

Only eval-mode, single-beam, identical unpadded-prefix batches with an ordinary
DynamicCache (or legacy tuple cache) are supported. Replay adds a second cached
forward trajectory. Traces live only until summarize(), in native dtype/device;
token reductions and optional operator products are batched by layer. Only scalar
rows leave this module. No sampling or file writing is performed.
"""

from __future__ import annotations

import copy
import inspect
import math
import weakref
from contextlib import ExitStack

import torch
from transformers.cache_utils import DynamicCache

__all__ = [
    "SteeringDiagnostics", "COMPLETION_METRICS", "LAYER_METRICS",
    "summarize_diagnostics",
]

COMPLETION_METRICS = (
    "n_tokens", "along_v_shift", "relative_change", "orthogonal_change",
    "kl_divergence", "normalized_push_energy", "mean_max_relative_push",
    "max_relative_push", "peak_push_layer",
)
LAYER_METRICS = (
    "layer", "n_tokens", "steered", "tracking_space", "beta", "beta_target",
    "beta_ratio", "relative_push", "relative_change", "orthogonal_change",
    "identity_model_error", "mean_jacobian_model_error", "dmd_model_error", "paired_model_error",
    "steering_dmd_model_error",
)
_EPS = 1e-8
_ACTIVE = weakref.WeakKeyDictionary()


def summarize_diagnostics(completion_rows):
    """Return equal-completion means and the modal peak layer (lowest index on ties)."""
    mean_fields = COMPLETION_METRICS[:-1]
    groups = {}
    for row in completion_rows:
        key = (row["concept"], row["method"], row["lambda"])
        if key not in groups:
            groups[key] = [0, dict.fromkeys(mean_fields, 0.0),
                           dict.fromkeys(mean_fields, 0), {}]
        n, sums, counts, peak_layers = groups[key]
        groups[key][0] = n + 1
        for field in mean_fields:
            value = row.get(field)
            if value is not None:
                sums[field] += value
                counts[field] += 1
        peak_layer = row.get("peak_push_layer")
        if peak_layer is not None:
            peak_layers[peak_layer] = peak_layers.get(peak_layer, 0) + 1
    return [
        {"concept": key[0], "method": key[1], "lambda": key[2], "n": n,
         **{field: sums[field] / counts[field] if counts[field] else None
            for field in mean_fields},
         "peak_push_layer": min(peak_layers, key=lambda layer: (-peak_layers[layer], layer))
         if peak_layers else None}
        for key, (n, sums, counts, peak_layers) in groups.items()
    ]


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype, torch.device)):
        return value
    raise ValueError(f"Unsupported forward argument type for exact replay: {type(value).__name__}.")


def _hidden(args, kwargs):
    hidden = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("Diagnostics require (batch, sequence, hidden) decoder states.")
    return hidden[:, -1].detach().clone()


def _logits_and_cache(output):
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "past_key_values", None)
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]
    raise ValueError("Diagnostics require causal-LM logits and returned past_key_values.")


def _hook_signature(modules):
    return tuple((tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                 for module in modules)


def _relative(numerator, denominator):
    return torch.where(denominator > _EPS, numerator / denominator.clamp_min(_EPS), float("nan"))


def _token_mean(values, valid):
    mask = valid.to(values.device) & torch.isfinite(values)
    counts = mask.sum(dim=0)
    sums = torch.where(mask, values, 0).sum(dim=0)
    return torch.where(counts > 0, sums / counts.clamp_min(1), float("nan"))


def _optional_rows(values):
    return [[value if math.isfinite(value) else None for value in row]
            for row in values.cpu().tolist()]


def _pooled_error(residual, denominator, valid):
    energy = torch.where(valid[:, :, None], residual.square(), 0).sum(dim=(0, 2))
    return torch.where(
        denominator > _EPS**2, (energy / denominator.clamp_min(_EPS**2)).sqrt(), float("nan"))


class SteeringDiagnostics:
    """Capture one generation batch, then summarize it after control hooks leave.

    Directions and optional raw dynamics stacks are retained by reference;
    dense operators move to a layer's device one matrix at a time during optional
    error calculation. All tracking and effect metrics use raw coordinates.
    """

    def __init__(self, model, contrastive_vecs, mean_jacobians=None, *,
                 dmd_dynamics=None, paired_dynamics=None, steering_dmd_dynamics=None,
                 model_errors=False, pair_sink=None):
        self.model = model
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.num_layers = len(self.layers)
        self.hidden_size = model.get_input_embeddings().weight.shape[1]
        shape = (self.num_layers + 1, self.hidden_size)
        if contrastive_vecs.shape != shape:
            raise ValueError(f"contrastive_vecs must have shape {shape}.")
        square = (self.num_layers, self.hidden_size, self.hidden_size)
        if mean_jacobians is not None and mean_jacobians.shape != square:
            raise ValueError("mean_jacobians must contain one raw square matrix per layer.")
        for name, stack in (("dmd_dynamics", dmd_dynamics), ("paired_dynamics", paired_dynamics),
                            ("steering_dmd_dynamics", steering_dmd_dynamics)):
            if stack is not None and stack.shape != square:
                raise ValueError(f"{name} must contain one raw A = I + B square matrix per layer.")
        self.directions = contrastive_vecs
        self.mean_jacobians = mean_jacobians
        self.dmd_dynamics = dmd_dynamics
        self.paired_dynamics = paired_dynamics
        self.steering_dmd_dynamics = steering_dmd_dynamics
        self.model_errors = model_errors
        self.pair_sink = pair_sink
        self._pending = None

    def record(self, controller=None, *, lmbda=0.0):
        """Observe the actual generation without installing or changing controls."""
        return _Capture(self, controller, lmbda)


class _Capture:
    def __init__(self, owner, controller, lmbda):
        self.owner = owner
        self.controller = controller
        self.lmbda = float(lmbda)
        self.steps = []
        self.handles = []
        self.current = None
        self.initial_cache = None
        self.entered = False
        self.exited = False
        self.used = False

    def __enter__(self):
        owner = self.owner
        previous = _ACTIVE.get(owner.model)
        if owner._pending is not None or (previous is not None and previous() is not None):
            raise RuntimeError("Overlapping or unsummarized diagnostics captures are not supported.")
        if owner.model.training:
            raise ValueError("Diagnostics require model.eval() for reproducible nominal replay.")
        if not math.isfinite(self.lmbda):
            raise ValueError("lmbda must be finite.")
        config = owner.model.generation_config
        if getattr(config, "num_beams", 1) not in (None, 1):
            raise ValueError("Diagnostics support single-beam generation only.")
        if self.controller is not None:
            if getattr(self.controller, "model", None) is not owner.model:
                raise ValueError("The diagnostics and controller must use the same model.")
            if getattr(self.controller, "hooks", ()):
                raise RuntimeError("Enter diagnostics before registering controller hooks.")
        self.selected = set(() if self.controller is None else self.controller.steered_layer_indices)
        if any(index not in range(owner.num_layers) for index in self.selected):
            raise ValueError("Controller selected an invalid decoder layer.")
        self.modules = (owner.model, *owner.layers, owner.norm)
        self.baseline_hooks = _hook_signature(self.modules)
        self.signature = inspect.signature(owner.model.forward)
        self.autocast = None
        owner._pending = self
        _ACTIVE[owner.model] = weakref.ref(self)
        self.entered = True
        try:
            self._install_hooks(actual=True)
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()
        self.current = None
        self.exited = True
        if exc_type is not None:
            self._cleanup()
        return False

    def _install_hooks(self, *, actual):
        owner = self.owner
        if actual:
            self.handles.append(owner.model.register_forward_pre_hook(
                self._forward_start, with_kwargs=True, prepend=True))
            self.handles.append(owner.model.register_forward_hook(
                self._forward_end, with_kwargs=True, prepend=True))
        for index, layer in enumerate(owner.layers):
            def before(module, args, kwargs, index=index):
                self.current["inputs"][index] = _hidden(args, kwargs)
            self.handles.append(layer.register_forward_pre_hook(
                before, with_kwargs=True, prepend=True))
            if actual:
                def after(module, args, output, index=index):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    self.current["pre_outputs"][index] = _hidden((hidden,), {})
                self.handles.append(layer.register_forward_hook(after, prepend=True))
        def final_state(module, args, kwargs):
            self.current["inputs"][-1] = _hidden(args, kwargs)
        self.handles.append(owner.norm.register_forward_pre_hook(
            final_state, with_kwargs=True, prepend=True))

    def _forward_start(self, module, args, kwargs):
        if self.current is not None:
            raise RuntimeError("Nested model forwards are unsupported by diagnostics.")
        bound = self.signature.bind(*args, **kwargs)
        arguments = {}
        for name, value in bound.arguments.items():
            kind = self.signature.parameters[name].kind
            if kind == inspect.Parameter.VAR_KEYWORD:
                arguments.update(value)
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError("Variadic positional forwards cannot be replayed exactly.")
            else:
                arguments[name] = value
        ids = arguments.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("Diagnostics require nonempty two-dimensional input_ids.")
        if arguments.get("inputs_embeds") is not None:
            raise ValueError("inputs_embeds generation is unsupported by diagnostics.")
        if not arguments.get("use_cache", getattr(module.config, "use_cache", False)):
            raise ValueError("Diagnostics require use_cache=True.")
        cache = arguments.pop("past_key_values", None)
        if cache is not None and type(cache) is not DynamicCache and not isinstance(cache, tuple):
            raise ValueError("Only ordinary DynamicCache or legacy tuple caches are supported.")
        if not self.steps:
            if cache is not None:
                length = cache.get_seq_length() if type(cache) is DynamicCache else (
                    cache[0][0].shape[-2] if cache else 0)
                if length:
                    raise ValueError("Generation must start with a fresh, empty cache.")
                # Copy BEFORE the actual forward mutates the cache, and never
                # keep a reference to the actual generation's KV trajectory.
                self.initial_cache = copy.deepcopy(cache)
            self.batch_size, self.prompt_width = ids.shape
        elif ids.shape != (self.batch_size, 1) or cache is None:
            raise ValueError("Diagnostics require one-token cached decoding after prefill.")
        autocast = tuple((device, torch.is_autocast_enabled(device),
                          torch.get_autocast_dtype(device)) for device in ("cpu", "cuda"))
        if self.autocast is None:
            self.autocast = autocast
        elif self.autocast != autocast:
            raise ValueError("Changing autocast mode during generation is unsupported.")
        step = {"kwargs": _snapshot(arguments), "inputs": [None] * (self.owner.num_layers + 1),
                "pre_outputs": [None] * self.owner.num_layers, "logits": None}
        self.steps.append(step)
        self.current = step

    def _forward_end(self, module, args, kwargs, output):
        logits, _ = _logits_and_cache(output)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("Diagnostics require raw (batch, sequence, vocabulary) logits.")
        if any(state is None for state in self.current["inputs"] + self.current["pre_outputs"]):
            raise ValueError("Not all decoder states were observed; this model layout is unsupported.")
        self.current["logits"] = logits[:, -1].detach().clone()
        self.current = None

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _cleanup(self):
        self._remove_hooks()
        self.steps.clear()
        self.current = None
        self.initial_cache = None
        self.controller = None
        if self.owner._pending is self:
            self.owner._pending = None
        active = _ACTIVE.get(self.owner.model)
        if active is not None and active() is self:
            del _ACTIVE[self.owner.model]

    def _validate_sequences(self, sequences):
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise ValueError("summarize requires the complete generated sequence tensor.")
        if not self.steps:
            raise ValueError("No generation forwards were captured.")
        if sequences.shape != (self.batch_size, self.prompt_width + len(self.steps)):
            raise ValueError("Captured forwards must match generated positions and batch size exactly.")
        prefix = self.steps[0]["kwargs"]["input_ids"]
        if not torch.equal(prefix, sequences[:, :self.prompt_width].to(prefix.device)):
            raise ValueError("Captured prompt IDs do not match supplied sequences.")
        if not torch.equal(prefix, prefix[:1].expand_as(prefix)):
            raise ValueError("Diagnostics require identical unpadded prefixes within a batch.")
        attention = self.steps[0]["kwargs"].get("attention_mask")
        if attention is not None and (not isinstance(attention, torch.Tensor)
                                     or attention.ndim != 2 or not bool(torch.all(attention == 1))):
            raise ValueError("Diagnostics support unpadded prefix attention masks only.")
        for index, step in enumerate(self.steps[1:], 1):
            ids = step["kwargs"]["input_ids"]
            expected = sequences[:, self.prompt_width + index - 1:self.prompt_width + index]
            if not torch.equal(ids, expected.to(ids.device)):
                raise ValueError("Captured consumed token IDs differ from the supplied generation.")
        eos_ids = self.owner.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = set()
        elif isinstance(eos_ids, torch.Tensor):
            eos_ids = set(eos_ids.detach().cpu().reshape(-1).tolist())
        elif isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids)
        generated = sequences[:, self.prompt_width:].detach().cpu().tolist()
        valid = torch.zeros(len(self.steps), self.batch_size, dtype=torch.bool)
        for batch_index, tokens in enumerate(generated):
            for token_index, token in enumerate(tokens):
                valid[token_index, batch_index] = True
                if token in eos_ids:
                    break
        return valid

    @torch.inference_mode()
    def summarize(self, sequences):
        """Replay without control and return (completion_rows, layer_rows), once."""
        if not self.entered or not self.exited or self.used:
            raise RuntimeError("summarize must be called once, after the recording context exits.")
        self.used = True
        try:
            if _hook_signature(self.modules) != self.baseline_hooks:
                raise RuntimeError("Remove controller hooks before nominal diagnostic replay.")
            valid = self._validate_sequences(sequences)
            devices = sorted({parameter.device.index for parameter in self.owner.model.parameters()
                              if parameter.device.type == "cuda"})
            with torch.random.fork_rng(devices=devices):
                with ExitStack() as stack:
                    for device, enabled, dtype in self.autocast:
                        stack.enter_context(torch.autocast(device, enabled=enabled, dtype=dtype))
                    nominal_states, kl = self._replay()
                with ExitStack() as stack:
                    for device, _, _ in self.autocast:
                        stack.enter_context(torch.autocast(device, enabled=False))
                    return self._reduce_metrics(nominal_states, kl, valid)
        finally:
            self._cleanup()

    def _replay(self):
        """Mirror generation's cached schedule; retain only states and scalar KL."""
        owner = self.owner
        nominal_states, divergences = [], []
        cache = self.initial_cache
        self.initial_cache = None
        self._install_hooks(actual=False)
        try:
            for actual in self.steps:
                nominal = {"inputs": [None] * (owner.num_layers + 1)}
                self.current = nominal
                kwargs = dict(actual["kwargs"])
                kwargs["past_key_values"] = cache
                output = owner.model(**kwargs)
                nominal_logits, cache = _logits_and_cache(output)
                if cache is None or any(state is None for state in nominal["inputs"]):
                    raise ValueError("Nominal replay did not produce the expected cache and states.")
                steered_logp = actual["logits"].float().log_softmax(dim=-1)
                nominal_logp = nominal_logits[:, -1].float().log_softmax(dim=-1)
                divergences.append((steered_logp.exp() * (steered_logp - nominal_logp)).sum(dim=-1))
                nominal_states.append(nominal["inputs"])
                actual["logits"] = None
                actual["kwargs"].clear()
                self.current = None
                del output, nominal_logits, steered_logp, nominal_logp, kwargs
        finally:
            self._remove_hooks()
            self.current = None
            cache = None
        return nominal_states, torch.stack(divergences)

    def _reduce_metrics(self, nominal_states, kl, valid):
        """Reduce all tokens together, visiting each decoder layer only once."""
        owner = self.owner
        n_tokens = valid.sum(dim=0).tolist()
        final_device = self.steps[0]["inputs"][-1].device
        push_energy = torch.zeros(valid.shape, device=final_device)
        maximum_push = torch.full(valid.shape, -float("inf"), device=final_device)
        peak_push = torch.full((self.batch_size,), -float("inf"), device=final_device)
        peak_layers = torch.full((self.batch_size,), -1, device=final_device, dtype=torch.long)
        final_valid = valid.to(final_device)
        pair_ids = None if owner.pair_sink is None else owner.pair_sink.reserve(self.batch_size)
        layer_rows = []
        for layer in range(owner.num_layers):
            x = torch.stack([step["inputs"][layer] for step in self.steps]).float()
            device = x.device
            nominal_x = torch.stack([states[layer] for states in nominal_states]).to(
                device=device, dtype=torch.float32)
            delta = x - nominal_x
            post_output = torch.stack([step["inputs"][layer + 1] for step in self.steps]).to(
                device=device, dtype=torch.float32)
            push = post_output - torch.stack([step["pre_outputs"][layer] for step in self.steps]).to(
                device=device, dtype=torch.float32)
            push_squared = push.square().sum(dim=-1)
            relative_push = _relative(push_squared.sqrt(), x.norm(dim=-1))
            nominal_norm = nominal_x.norm(dim=-1)
            relative_change = _relative(delta.norm(dim=-1), nominal_norm)
            direction = owner.directions[layer].detach().to(device=device, dtype=torch.float32)
            norm = direction.norm()
            orthogonal_change = torch.full_like(relative_change, float("nan"))
            if norm > _EPS:
                raw_direction = direction / norm
                orthogonal = delta - (delta @ raw_direction)[:, :, None] * raw_direction
                orthogonal_change = _relative(orthogonal.norm(dim=-1), nominal_norm)
            beta = torch.full_like(relative_change, float("nan"))
            target = None
            if norm > _EPS:
                beta = x @ raw_direction
                target = self.lmbda * norm.item()
            beta_mean = _token_mean(beta, valid)
            beta_ratio = beta_mean / target if target is not None and abs(target) > _EPS else torch.full_like(beta_mean, float("nan"))
            identity_error = torch.full_like(beta_mean, float("nan"))
            jacobian_error = torch.full_like(beta_mean, float("nan"))
            dmd_error = torch.full_like(beta_mean, float("nan"))
            paired_error = torch.full_like(beta_mean, float("nan"))
            steering_error = torch.full_like(beta_mean, float("nan"))
            delta_next = None
            if owner.model_errors or owner.pair_sink is not None:
                delta_next = post_output - torch.stack([states[layer + 1] for states in nominal_states]).to(
                    device=device, dtype=torch.float32)
            if owner.model_errors:
                mask = valid.to(device)
                denominator = torch.where(mask[:, :, None], delta_next.square(), 0).sum(dim=(0, 2))
                identity_error = _pooled_error(delta_next - (delta + push), denominator, mask)
                if owner.mean_jacobians is not None:
                    jacobian = owner.mean_jacobians[layer].detach().to(device=device, dtype=torch.float32)
                    jacobian_error = _pooled_error(delta_next - (delta @ jacobian.T + push), denominator, mask)
                    del jacobian
                if owner.dmd_dynamics is not None:
                    dynamics = owner.dmd_dynamics[layer].detach().to(device=device, dtype=torch.float32)
                    dmd_error = _pooled_error(delta_next - (delta @ dynamics.T + push), denominator, mask)
                    del dynamics
                if owner.paired_dynamics is not None:
                    dynamics = owner.paired_dynamics[layer].detach().to(device=device, dtype=torch.float32)
                    paired_error = _pooled_error(delta_next - (delta @ dynamics.T + push), denominator, mask)
                    del dynamics
                if owner.steering_dmd_dynamics is not None:
                    dynamics = owner.steering_dmd_dynamics[layer].detach().to(device=device, dtype=torch.float32)
                    steering_error = _pooled_error(delta_next - (delta @ dynamics.T + push), denominator, mask)
                    del dynamics
            if owner.pair_sink is not None:
                owner.pair_sink.add(layer, delta, push, delta_next, valid, pair_ids)
            del delta_next
            values = torch.stack((
                beta_mean, beta_ratio, _token_mean(relative_push, valid),
                _token_mean(relative_change, valid), _token_mean(orthogonal_change, valid),
                identity_error, jacobian_error, dmd_error, paired_error, steering_error,
            ), dim=-1)
            for batch_index, (beta_value, ratio, effort, change, orthogonal, error_i, error_j, error_dmd, error_paired, error_steering) in enumerate(_optional_rows(values)):
                layer_rows.append({
                    "batch_index": batch_index, "layer": layer, "n_tokens": n_tokens[batch_index],
                    "steered": layer in self.selected, "tracking_space": "raw",
                    "beta": beta_value, "beta_target": target, "beta_ratio": ratio,
                    "relative_push": effort, "relative_change": change, "orthogonal_change": orthogonal,
                    "identity_model_error": error_i, "mean_jacobian_model_error": error_j,
                    "dmd_model_error": error_dmd, "paired_model_error": error_paired,
                    "steering_dmd_model_error": error_steering,
                })
            push_energy += relative_push.square().to(final_device)
            candidate = torch.where(torch.isfinite(relative_push), relative_push, -float("inf")).to(final_device)
            maximum_push = torch.maximum(maximum_push, candidate)
            layer_peak = torch.where(final_valid, candidate, -float("inf")).amax(dim=0)
            peak_layers = torch.where(layer_peak > peak_push, layer, peak_layers)
            peak_push = torch.maximum(peak_push, layer_peak)
            for actual, nominal in zip(self.steps, nominal_states):
                actual["inputs"][layer] = None
                actual["pre_outputs"][layer] = None
                nominal[layer] = None
        final_actual = torch.stack([step["inputs"][-1] for step in self.steps]).float()
        final_nominal = torch.stack([states[-1] for states in nominal_states]).to(
            device=final_device, dtype=torch.float32)
        delta_final = final_actual - final_nominal
        final_norm = final_nominal.norm(dim=-1)
        along = torch.full_like(final_norm, float("nan"))
        orthogonal_change = torch.full_like(final_norm, float("nan"))
        direction = owner.directions[-1].detach().to(device=final_device, dtype=torch.float32)
        norm = direction.norm()
        if norm > _EPS:
            direction = direction / norm
            along = delta_final @ direction
            orthogonal = delta_final - along[:, :, None] * direction
            orthogonal_change = _relative(orthogonal.norm(dim=-1), final_norm)
        peak_layer_rows = peak_layers.cpu().tolist()
        values = torch.stack((
            _token_mean(along, valid), _token_mean(_relative(delta_final.norm(dim=-1), final_norm), valid),
            _token_mean(orthogonal_change, valid), _token_mean(kl.to(final_device), valid),
            _token_mean(push_energy, valid), _token_mean(maximum_push, valid), peak_push,
        ), dim=-1)
        completion_rows = [
            {"batch_index": index, "n_tokens": n_tokens[index],
             **dict(zip(COMPLETION_METRICS[1:-1], row)),
             "peak_push_layer": peak_layer_rows[index] if row[-1] is not None and row[-1] > 0 else None}
            for index, row in enumerate(_optional_rows(values))
        ]
        return completion_rows, layer_rows
