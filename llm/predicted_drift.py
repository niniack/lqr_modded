"""Forward-local predicted drift cancellation, with optional nominal diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from functools import partial
from inspect import signature

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache, DynamicCache

from .model import blocks
from .steering import LQRSteering, riccati


def _clone_jvp_inputs(value, layer):
    """Detach inference tensors and isolate only the cache layer this block updates."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Cache):
        if value.offloading:
            raise ValueError("Local Jacobian predictors require a non-offloaded KV cache")
        cloned = copy(value)
        cloned.layers = list(value.layers)
        if layer < len(value.layers):
            cloned.layers[layer] = copy(value.layers[layer])
            for name, item in vars(value.layers[layer]).items():
                setattr(cloned.layers[layer], name, _clone_jvp_inputs(item, layer))
        return cloned
    if isinstance(value, dict):
        return {key: _clone_jvp_inputs(item, layer) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_jvp_inputs(item, layer) for item in value)
    if isinstance(value, list):
        return [_clone_jvp_inputs(item, layer) for item in value]
    return value


class PredictedDriftLQRSteering(LQRSteering):
    """Apply alpha K v - drift_weight K d_perp, predicting d_next = P d + u.

    Gains come from A, independently of predictor P. A tensor predictor uses the
    original fixed-matrix update. "local_jacobian" instead uses a batched JVP at
    the actual block inputs; "local_jacobian_nom" uses the independent nominal
    model's inputs and cache. Each perturbs one position, holding all other
    positions and cached keys/values fixed. JVPs use the model's dtype and math
    attention on detached input/cache copies, outside inference mode.
    Derivative calls bypass block hooks and use pre-update cache snapshots.

    The FP32 deviation starts at zero on every forward: this approximation models
    neither cross-token coupling nor deviations carried in the decode KV history.
    With weight zero, the command and native-dtype addition are the parent's.

    drift_log holds (layer, ||predicted d_perp|| / ||x||) at steered block inputs.
    Optional nominal diagnostics compare projected predicted and measured drift
    before the current push. prediction_log entries contain step (zero-based
    forward index), layer, relative_error, reference_norm (the measured projected
    drift norm), measured_relative_drift, predicted_relative_drift, and cosine_perp.
    The relative drift norms divide by the current steered input norm; tensors
    have shape (batch, selected positions). Zero denominators give NaN. Logs reset
    on active(). Full-sequence replay computes token-diagonal JVPs separately for
    each selected position, with the entire batch in each call.

    Outside active(), push() returns the latest forward's stored FP32 commands,
    not native-dtype rounded increments. Consume trace metrics before reusing
    the controller; shape checks cannot distinguish same-shaped older traces.
    nominal must be an independent identical eval-mode model. It only measures
    prediction quality, except when explicitly selecting "local_jacobian_nom".
    """

    def __init__(self, model, A, E, q, r, qf, predictor, drift_weight=0.0,
                 nominal=None, **kwargs):
        if nominal is not None and (nominal is model or nominal.base_model is model.base_model):
            raise ValueError("nominal must be a separate, unsteered model instance")
        self._local_predictor = predictor if isinstance(predictor, str) else None
        if self._local_predictor not in (None, "local_jacobian", "local_jacobian_nom"):
            raise ValueError(f"Unknown local predictor: {predictor!r}")
        if self._local_predictor == "local_jacobian_nom" and nominal is None:
            raise ValueError("local_jacobian_nom requires an independent nominal model")
        super().__init__(model, A, E, q, r, qf, **kwargs)
        devices = [next(layer.parameters()).device for layer in self.layers]
        self._P = None
        if self._local_predictor is None:
            expected = (len(self.layers), E.shape[-1], E.shape[-1])
            if tuple(predictor.shape) != expected:
                raise ValueError(f"predictor must have shape {expected}, got {tuple(predictor.shape)}")
            self._P = [p.detach().to(device=d, dtype=torch.float32)
                       for p, d in zip(predictor, devices)]
        self._K = {
            layer: K.detach().to(device=devices[layer], dtype=torch.float32)
            for layer, K in riccati(A, q, r, qf, devices[0], self.steered)
            if layer in self.steered
        }
        self.drift_weight = drift_weight
        self.nominal = nominal
        self.nominal_layers = [] if nominal is None else list(blocks(nominal))
        if nominal is not None and len(self.nominal_layers) != len(self.layers):
            raise ValueError("steered and nominal models must have matching decoder stacks")
        self._forward_signature = signature(model.base_model.forward)
        self._active = False
        self._ready = False
        self._d = None
        self._controls = {}
        self._x_nom = {}
        self._linearizations = {}
        self._cache = None
        self._step = -1
        self.drift_log = []
        self.prediction_log = []

    def push(self, layer, x):
        """Compute an active command, or retrieve a completed forward for replay."""
        if not self._active:
            if not self._ready or layer not in self._controls:
                raise RuntimeError("No completed control for this layer; run a steered forward before replay metrics")
            u = self._controls[layer]
            if u.shape != x.shape:
                raise ValueError(
                    f"Stored control shape {tuple(u.shape)} does not match replay input {tuple(x.shape)}; "
                    "consume metrics immediately after the corresponding trace, before controller reuse"
                )
            return u.to(x.device)
        if self._d is None or self._d.shape != x.shape:
            raise RuntimeError("Active push requires the current block's forward-local predicted deviation")
        u = super().push(layer, x)
        v = self._v[layer].to(x.device)
        d_perp = self._d - (self._d @ v)[..., None] * v
        input_norm = x.float().norm(dim=-1)
        predicted_norm = d_perp.norm(dim=-1)
        predicted_relative = torch.where(input_norm > 0, predicted_norm / input_norm, torch.nan)
        self.drift_log.append((layer, predicted_relative.detach()))
        if self.nominal is not None:
            measured = x.float() - self._x_nom[layer].to(x.device)
            measured_perp = measured - (measured @ v)[..., None] * v
            reference = measured_perp.norm(dim=-1)
            error = (d_perp - measured_perp).norm(dim=-1)
            relative = torch.where(reference > 0, error / reference, torch.nan)
            cosine_denominator = predicted_norm * reference
            cosine = torch.where(
                cosine_denominator > 0, (d_perp * measured_perp).sum(-1) / cosine_denominator, torch.nan,
            )
            self.prediction_log.append({
                "step": self._step,
                "layer": layer,
                "relative_error": relative.detach(),
                "reference_norm": reference.detach(),
                "measured_relative_drift": torch.where(input_norm > 0, reference / input_norm, torch.nan).detach(),
                "predicted_relative_drift": predicted_relative.detach(),
                "cosine_perp": cosine.clamp(-1, 1).detach(),
            })
        if self.drift_weight == 0:
            return u
        return u - self.drift_weight * (d_perp @ self._K[layer].T)

    def _record_linearization(self, layer, module, args, kwargs):
        # Capture BEFORE the real block appends its current keys and values.
        with torch.inference_mode(False), torch.no_grad():
            self._linearizations[layer] = (
                _clone_jvp_inputs(args, layer), _clone_jvp_inputs(kwargs, layer),
            )

    def _local_jvp(self, layer, direction):
        args, kwargs = self._linearizations.pop(layer)
        block = self.nominal_layers[layer] if self._local_predictor == "local_jacobian_nom" else self.layers[layer]
        hidden = args[0] if args else kwargs["hidden_states"]
        positions = range(*slice(self.push_from, None).indices(hidden.shape[1]))
        with torch.inference_mode(False), torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            tangent = direction.detach().to(device=hidden.device, dtype=hidden.dtype).clone()
            propagated = torch.empty_like(tangent, dtype=torch.float32)
            for selected, position in enumerate(positions):
                def last_output(last):
                    inputs = torch.cat((hidden[:, :position], last[:, None], hidden[:, position + 1:]), dim=1)
                    call_args = tuple(_clone_jvp_inputs(item, layer) if isinstance(item, Cache) else item
                                      for item in args)
                    call_kwargs = {key: _clone_jvp_inputs(value, layer) if isinstance(value, Cache) else value
                                   for key, value in kwargs.items()}
                    if call_args:
                        call_args = (inputs, *call_args[1:])
                    else:
                        call_kwargs["hidden_states"] = inputs
                    # Bypass block hooks: this is a pure derivative call, not another control step.
                    output = block.forward(*call_args, **call_kwargs)
                    output = output[0] if isinstance(output, (tuple, list)) else output
                    return output[:, position]

                propagated[:, selected] = torch.func.jvp(
                    last_output, (hidden[:, position],), (tangent[:, selected],),
                )[1].float()
        return propagated.to(direction.device)

    def _hook(self, layer, module, args, kwargs, output):
        x = (args[0] if args else kwargs["hidden_states"])[:, self.push_from:]
        if layer == 0:
            self._step += 1
            self._d = torch.zeros_like(x, dtype=torch.float32)
            self._controls = {}
            self._ready = False
        else:
            self._d = self._d.to(x.device)
        u = self.push(layer, x) if layer in self.steered else None
        if self._local_predictor is None:
            self._d = (self._d @ self._P[layer].T).detach()
        else:
            self._d = self._local_jvp(layer, self._d).detach()
        if u is not None:
            self._controls[layer] = u.detach()
            self._d = self._d + self._controls[layer]
        if layer == len(self.layers) - 1:
            self._ready = True
            self._x_nom = {}
        if u is None:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[:, self.push_from:] += u.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

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
        self._x_nom = {}
        # Do not retain nominal outputs; only detached block inputs are needed.
        self.nominal.base_model(**inputs)

    def _record_nominal(self, layer, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        self._x_nom[layer] = hidden[:, self.push_from:].detach().float().clone()

    @contextmanager
    def active(self, lmbda, push_from=-1):
        """Hook all blocks; preserve completed commands, but release nominal caches."""
        if self._active:
            raise RuntimeError("PredictedDriftLQRSteering.active() contexts cannot be nested")
        self.lmbda, self.push_from = lmbda, push_from
        self._d, self._cache = None, None
        self._controls, self._x_nom = {}, {}
        self._linearizations = {}
        self._step, self._ready = -1, False
        self.drift_log, self.prediction_log = [], []
        handles = []
        try:
            self._active = True
            for i, layer in enumerate(self.layers):
                if self._local_predictor == "local_jacobian":
                    handles.append(layer.register_forward_pre_hook(
                        partial(self._record_linearization, i), with_kwargs=True,
                    ))
                handles.append(layer.register_forward_hook(partial(self._hook, i), with_kwargs=True))
            if self.nominal is not None:
                if self._local_predictor == "local_jacobian_nom":
                    for i, layer in enumerate(self.nominal_layers):
                        handles.append(layer.register_forward_pre_hook(
                            partial(self._record_linearization, i), with_kwargs=True,
                        ))
                for i in self.steered:
                    handles.append(self.nominal_layers[i].register_forward_pre_hook(
                        partial(self._record_nominal, i), with_kwargs=True,
                    ))
                handles.append(self.model.base_model.register_forward_pre_hook(
                    self._run_nominal, with_kwargs=True,
                ))
            yield self
        except BaseException:
            self._controls, self._ready = {}, False
            raise
        finally:
            for handle in handles:
                handle.remove()
            self._active = False
            self._d, self._cache = None, None
            self._x_nom = {}
            self._linearizations = {}
