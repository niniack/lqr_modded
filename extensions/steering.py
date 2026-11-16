from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
from inspect import signature

import torch
from transformers.cache_utils import DynamicCache

from steer.lqr_utils import time_varying_lqr_noB_mem_efficient
from steer.steering import LQRSteering as UpstreamLQRSteering, Mode

from .data import LayerInputs


def riccati(A, q, r, qf, device="cpu", start_layer=0):
    """Stream float64 LQR gains for a contiguous suffix in descending layer order.

    Upstream solves each selected stage with B=I. Its API returns gains, not
    cost-to-go, so we carry the latter across one-stage calls. This avoids
    retaining O(layers * hidden**2) upstream workspaces. Earlier uncontrolled
    stages cannot affect suffix gains and are not evaluated.
    The cost update and symmetrization retain the frozen solver's ordering.
    """
    hidden = A.shape[-1]
    eye = torch.eye(hidden, dtype=torch.float64, device=device)
    Q, R = (q * eye).unsqueeze(0), (r * eye).unsqueeze(0)
    S = qf * eye
    for layer in reversed(range(start_layer, A.shape[0])):
        a = A[layer].to(device, torch.float64)
        F = S @ a
        K = time_varying_lqr_noB_mem_efficient(a.unsqueeze(0), Q, R, S)[0]
        yield layer, K
        S = Q[0] + a.T @ F - F.T @ K
        S = (S + S.T) / 2


class QwenLQRSteering(UpstreamLQRSteering):
    """Qwen input/dtype bridge to the original upstream linear setpoint controller.

    The original hook computes every base push, using full FP32 K on selected
    B×T×H states. This differs in rounding from the frozen implementation's
    float64-precomputed Kv followed by scalar multiplication; cached results
    therefore are not a claim of bitwise equality with this controller.

    Only active(), push(), and sample() are supported here, not the upstream
    angular/multi-concept generation workflows. E_unit holds one FP32 vector
    per state boundary on that boundary's device; K holds actuated layers only.
    """

    def __init__(self, model, A, E, q, r, qf, steer_last_n_layers=None):
        # Do not call the upstream constructor: even A=None allocates dense
        # per-layer Q/R before discarding them. Initialize only the state used
        # by its setpoint hook, registration, and context lifecycle instead.
        self.model = model
        self.layers = list(model.model.layers)
        self.T = len(self.layers)
        if steer_last_n_layers is not None and not 1 <= steer_last_n_layers <= self.T:
            raise ValueError(f"steer_last_n_layers must be between 1 and {self.T}")
        start = 0 if steer_last_n_layers is None else self.T - steer_last_n_layers
        self.steered = range(start, self.T)
        devices = [next(layer.parameters()).device for layer in self.layers]
        devices.append(next(model.model.norm.parameters()).device)
        self.device = devices[0]
        self.E = E.float()
        norms = self.E.norm(dim=-1)
        self._norms = norms.tolist()
        self.E_unit = [
            (e / norm.clamp_min(1e-12)).to(device)
            for e, norm, device in zip(self.E, norms, devices)
        ]
        self.K = {}
        for layer, K in riccati(A, q, r, qf, self.device, start_layer=start):
            self.K[layer] = K.to(device=devices[layer], dtype=torch.float32)
        # Real upstream logging buffers copy the final state/control vector,
        # rather than keeping views that retain entire replay activations.
        self.X = torch.empty(self.T, E.shape[-1], device=self.device, dtype=torch.float32)
        self.U = torch.empty_like(self.X)
        self.hooks = []
        self.mode = Mode.SETPOINT
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = False
        # push() always receives a selected B×T×H slice, including T=1.
        self.ALL_TOKENS = True
        self.lmbda = 0.0
        self.push_from = -1
        self._active = False

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._lmbda = value
        self.betas = [value * norm for norm in self._norms]

    def push(self, layer, x):
        """Unquantized FP32 control for selected block-input states (B, T, H)."""
        x = x.float()
        if layer not in self.K:
            return torch.zeros_like(x)
        self.device = x.device
        # A zero FP32 output extracts the original hook's actual u without
        # quantizing it into model output. Drift correction and effort metrics
        # then use that same control; no second base feedback formula exists.
        return super().hook_setpoint_tracking(
            layer, self.layers[layer], (x,), torch.zeros_like(x),
        )

    @staticmethod
    def _positional_hidden(module, args, kwargs):
        # Current Transformers supplies hidden_states by keyword, whereas
        # upstream's unchanged hook registration cannot request with_kwargs.
        # Keep model precision here; only the control calculation uses FP32.
        if not args:
            return (kwargs.pop("hidden_states"),), kwargs

    def hook_setpoint_tracking(self, layer_idx, module, input, output):
        if layer_idx not in self.steered:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        x = input[0][:, self.push_from:]
        hidden = hidden.clone()
        hidden[:, self.push_from:] += self.push(layer_idx, x).to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    @contextmanager
    def active(self, lmbda, push_from=-1):
        """Push selected positions; -1 steers only the last token in generation."""
        if self._active:
            raise RuntimeError("Controller active() contexts cannot be nested")
        self.lmbda, self.push_from = lmbda, push_from
        self._active = True
        try:
            for i in self.steered:
                self.hooks.append(self.layers[i].register_forward_pre_hook(
                    self._positional_hidden, with_kwargs=True,
                ))
            # Inherited __enter__ registers the original hook wrappers on all
            # blocks; our hook skips unactuated ones. __exit__ removes both the
            # wrappers and the positional bridges via inherited remove_hooks.
            with self:
                yield self
        finally:
            if self.hooks:  # also clean up if registration itself raised
                self.remove_hooks()
            self._active = False


class DriftLQRSteering(QwenLQRSteering):
    """A-LQR plus -drift_weight K_l d_perp, relative to an independent nominal model.

    Both models must have identical weights/configuration and be in eval mode. The nominal
    model follows the same tokens, with a separate cache. Hooks attach to the base model
    so cached generation and the existing full-sequence trace() both work.

    drift_log contains (layer, |d|/|x_nom|, |d_perp|/|x_nom|) tensors at block inputs.
    Each active() resets the log; dry_run logs drift but applies no control. At zero
    drift_weight the push is exactly the parent's. Full gains are retained only for
    actuated blocks. Evaluate replay metrics before reusing/changing the controller.
    """

    def __init__(self, model, nominal, A, E, q, r, qf, drift_weight=0.0,
                 steer_last_n_layers=None):
        if nominal is model or nominal.base_model is model.base_model:
            raise ValueError("nominal must be a separate, unsteered model instance")
        super().__init__(model, A, E, q, r, qf, steer_last_n_layers=steer_last_n_layers)
        self.nominal = nominal
        self.nominal_layers = list(nominal.model.layers)
        if len(self.nominal_layers) != len(self.layers):
            raise ValueError("steered and nominal models must have matching decoder stacks")
        self.drift_weight = drift_weight
        self.dry_run = False
        self._forward_signature = signature(model.base_model.forward)
        self._x_nom = {}
        self._cache = None
        self.drift_log = []

    def push(self, layer, x):
        u = super().push(layer, x)
        x_nom = self._x_nom[layer].to(x.device)
        v = self.E_unit[layer].to(x.device)
        d = x.float() - x_nom
        d_perp = d - (d @ v)[..., None] * v
        if self._active:
            size = x_nom.norm(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
            self.drift_log.append((
                layer, (d.norm(dim=-1) / size).detach(),
                (d_perp.norm(dim=-1) / size).detach(),
            ))
        if self.dry_run:
            return torch.zeros_like(u)
        if self.drift_weight == 0:
            return u
        return u - self.drift_weight * (d_perp @ self.K[layer].T)

    @torch.no_grad()
    def _run_nominal(self, module, args, kwargs):
        inputs = dict(self._forward_signature.bind_partial(*args, **kwargs).arguments)
        inputs.update(inputs.pop("kwargs", {}))
        cache = inputs.get("past_key_values")
        use_cache = inputs.get("use_cache")
        if use_cache is None:
            use_cache = module.config.use_cache
        if use_cache:
            seen = 0 if cache is None else cache.get_seq_length()
            if seen == 0:
                self._cache = DynamicCache(config=self.nominal.config)
            elif self._cache is None or self._cache.get_seq_length() != seen:
                raise ValueError("Start generation with an empty cache inside active(); nominal cache is out of sync")
        else:
            self._cache = None
        inputs["past_key_values"] = self._cache
        inputs["use_cache"] = use_cache
        self.nominal.base_model(**inputs)

    def _record(self, layer, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        self._x_nom[layer] = hidden[:, self.push_from:].detach().float().clone()

    @contextmanager
    def active(self, lmbda, push_from=-1):
        if self._active:
            raise RuntimeError("DriftLQRSteering.active() contexts cannot be nested")
        self._x_nom, self._cache, self.drift_log = {}, None, []
        handles = []
        try:
            with super().active(lmbda, push_from):
                for i in self.steered:
                    handles.append(self.nominal_layers[i].register_forward_pre_hook(
                        partial(self._record, i), with_kwargs=True,
                    ))
                handles.append(self.model.base_model.register_forward_pre_hook(
                    self._run_nominal, with_kwargs=True,
                ))
                yield self
        finally:
            for handle in handles:
                handle.remove()
            self._cache = None
            # Keep replay reference states for completion_metrics().


@torch.inference_mode()
def sample(model, tokenizer, prompt, n, *, controller=None, lmbda=0.0, max_new_tokens=96,
           temperature=0.7, top_p=0.95):
    """n sampled completions of one prompt, steered if a controller is given. Returns token ids."""
    inputs = tokenizer([prompt] * n, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p,
                  repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
    with controller.active(lmbda) if controller else nullcontext():
        return model.generate(**inputs, **kwargs)


@torch.inference_mode()
def trace(model, controller, lmbda, sequences, prompt_len):
    """Replay the same generated tokens with and without control.

    Returns boundary states (L + 1, B, T, H) and a valid (B, T) mask through
    the first EOS. Pushing from the last prompt token reproduces generation's
    last-token pushes in one teacher-forced pass.
    """
    ids = sequences[:, :-1]
    start = prompt_len - 1
    with LayerInputs(model) as record, controller.active(lmbda, push_from=start):
        model.base_model(input_ids=ids, use_cache=False)
    steered = torch.stack([state[:, start:] for state in record.states])
    with LayerInputs(model) as record:
        model.base_model(input_ids=ids, use_cache=False)
    nominal = torch.stack([state[:, start:] for state in record.states])
    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=sequences.device)
    is_eos = torch.isin(sequences[:, prompt_len:], eos).long()
    valid = (is_eos.cumsum(dim=1) - is_eos) == 0
    return {"steered": steered, "nominal": nominal, "valid": valid,
            "controller": controller, "lmbda": lmbda}


def _token_mean(values, valid):
    keep = valid & values.isfinite()
    return torch.where(keep, values, 0).sum(dim=1) / keep.sum(dim=1)


def completion_metrics(tr, *, per_layer=False):
    """Per-completion token means at the final pre-norm state.

    along_v: v^T dx; change: |dx| / |x_unsteered|;
    orthogonal: |dx off v| / |x_unsteered|;
    push_energy: sum_l |u_l|^2 / |x_l|^2, using unquantized FP32 controls.
    With per_layer=True, layer_push_energy contains the token-mean energy at
    each block, with zeros for the uncontrolled prefix.
    """
    valid = tr["valid"]
    v = tr["controller"].E_unit[-1].to(tr["steered"].device)
    dx = tr["steered"][-1].float() - tr["nominal"][-1].float()
    reference = tr["nominal"][-1].float().norm(dim=-1)
    along = dx @ v
    off = dx - along[..., None] * v
    energy = torch.zeros(valid.shape, device=valid.device)
    controller = tr["controller"]
    layer_energy = energy.new_zeros(len(valid), controller.T) if per_layer else None
    controller.lmbda = tr["lmbda"]
    for layer in controller.steered:
        x = tr["steered"][layer].float()
        u = controller.push(layer, x)
        contribution = u.square().sum(-1) / x.square().sum(-1)
        energy += contribution
        if per_layer:
            layer_energy[:, layer] = _token_mean(contribution, valid)
    columns = {
        "along_v": _token_mean(along, valid),
        "change": _token_mean(dx.norm(dim=-1) / reference, valid),
        "orthogonal": _token_mean(off.norm(dim=-1) / reference, valid),
        "push_energy": _token_mean(energy, valid),
    }
    rows = [{name: values[i].item() for name, values in columns.items()} for i in range(len(valid))]
    if per_layer:
        for row, values in zip(rows, layer_energy.cpu().tolist()):
            row["layer_push_energy"] = values
    return rows


def alignment_metrics(tr):
    """Token-mean shares of squared final-state change; zero changes have no direction."""
    delta = tr["steered"][-1].float() - tr["nominal"][-1].float()
    v = tr["controller"].E_unit[-1].to(delta.device)
    along = delta @ v
    squared_change = delta.square().sum(-1)
    along_fraction = (along.square() / squared_change).clamp(0, 1)
    columns = {
        "change": _token_mean(squared_change.sqrt() / tr["nominal"][-1].float().norm(dim=-1), tr["valid"]),
        "along_v": _token_mean(along, tr["valid"]),
        "along_fraction": _token_mean(along_fraction, tr["valid"]),
        "off_fraction": _token_mean(1 - along_fraction, tr["valid"]),
    }
    return [{name: values[i].item() for name, values in columns.items()} for i in range(len(along))]