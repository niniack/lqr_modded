from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
from inspect import signature

import torch
from transformers.cache_utils import DynamicCache

from .model import blocks


def riccati(A, q, r, qf, device="cpu", actuated=None):
    """Finite-horizon LQR gains for e_{l+1} = A_l e_l - u_l, u_l = K_l e_l, with Q = qI, R = rI, Qf = qf I.

    actuated: blocks that may push (None = all). Elsewhere K_l = 0 and S_l = qI + A_l^T S_{l+1} A_l,
    so a push plans across the free blocks after it with the model.
    Yields (layer, K_l) from the last layer to the first; K_l is float64 on `device`.
    """
    hidden = A.shape[-1]
    eye = torch.eye(hidden, dtype=torch.float64, device=device)
    S = qf * eye
    for layer in reversed(range(A.shape[0])):
        a = A[layer].to(device, torch.float64)
        F = S @ a
        K = torch.linalg.solve(S + r * eye, F) if actuated is None or layer in actuated else torch.zeros_like(a)
        yield layer, K
        S = q * eye + a.T @ F - F.T @ K
        S = (S + S.T) / 2


class LQRSteering:
    """A-LQR: after block l, push the last token by u_l = (lmbda ||E_l|| - v_l^T x_l) K_l v_l.

    x_l is the token's state entering block l and v_l = E_l / ||E_l||. Only K_l v_l is
    needed, so the full gain matrices are never stored. steer_last_n_layers restricts actuation
    to the final N blocks; every = m selects every m-th block within that suffix.
    """

    def __init__(self, model, A, E, q, r, qf, every=1, steer_last_n_layers=None):
        self.model = model
        self.layers = list(blocks(model))
        num_layers = len(self.layers)
        if steer_last_n_layers is not None and not 1 <= steer_last_n_layers <= num_layers:
            raise ValueError(f"steer_last_n_layers must be between 1 and {num_layers}")
        start = 0 if steer_last_n_layers is None else num_layers - steer_last_n_layers
        self.steered = range(start, num_layers, every)
        self.size = E.float().norm(dim=-1)
        self.v = E.float() / self.size.clamp_min(1e-12)[:, None]
        self.gain = torch.empty(num_layers, E.shape[-1])
        device = next(self.layers[0].parameters()).device
        for layer, K in riccati(A, q, r, qf, device, self.steered):
            self.gain[layer] = (K @ self.v[layer].to(device, torch.float64)).float().cpu()
        devices = [next(layer.parameters()).device for layer in self.layers]
        self._v = [self.v[i].to(d) for i, d in enumerate(devices)]
        self._gain = [self.gain[i].to(d) for i, d in enumerate(devices)]
        self._size = self.size.tolist()
        self.lmbda = 0.0
        self.push_from = -1

    def push(self, layer, x):
        """Control for states x (..., H) entering `layer`, in float32."""
        v, gain = self._v[layer].to(x.device), self._gain[layer].to(x.device)
        error = self.lmbda * self._size[layer] - x.float() @ v
        return error[..., None] * gain

    def _hook(self, layer, module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        x = (args[0] if args else kwargs["hidden_states"])[:, self.push_from:]
        hidden = hidden.clone()
        hidden[:, self.push_from:] += self.push(layer, x).to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    @contextmanager
    def active(self, lmbda, push_from=-1):
        """Steer while inside the block. push_from=-1 pushes only the last position (generation)."""
        self.lmbda, self.push_from = lmbda, push_from
        handles = [self.layers[i].register_forward_hook(partial(self._hook, i), with_kwargs=True)
                   for i in self.steered]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()


class DriftLQRSteering(LQRSteering):
    """A-LQR plus -drift_weight K_l d_perp, relative to an independent nominal model.

    Both models must have identical weights/configuration and be in eval mode. The nominal
    model follows the same tokens, with a separate cache. Hooks attach to the base model
    so cached generation and the existing full-sequence trace() both work.

    drift_log contains (layer, |d|/|x_nom|, |d_perp|/|x_nom|) tensors at block inputs.
    Each active() resets the log; dry_run logs drift but applies no control. At zero
    drift_weight the push is exactly the parent's. Full gains are retained only for
    actuated blocks. Evaluate replay metrics before reusing/changing the controller.
    """

    def __init__(self, model, nominal, A, E, q, r, qf, drift_weight=0.0, every=1,
                 steer_last_n_layers=None):
        if nominal is model or nominal.base_model is model.base_model:
            raise ValueError("nominal must be a separate, unsteered model instance")
        super().__init__(model, A, E, q, r, qf, every=every, steer_last_n_layers=steer_last_n_layers)
        self.nominal = nominal
        self.nominal_layers = list(blocks(nominal))
        if len(self.nominal_layers) != len(self.layers):
            raise ValueError("steered and nominal models must have matching decoder stacks")
        self.drift_weight = drift_weight
        self.dry_run = False
        devices = [next(layer.parameters()).device for layer in self.layers]
        self._K = {
            layer: K.float().to(devices[layer])
            for layer, K in riccati(A, q, r, qf, devices[0], self.steered)
            if layer in self.steered
        }
        self._forward_signature = signature(model.base_model.forward)
        self._x_nom = {}
        self._cache = None
        self._active = False
        self.drift_log = []

    def push(self, layer, x):
        u = super().push(layer, x)
        x_nom = self._x_nom[layer].to(x.device)
        v = self._v[layer].to(x.device)
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
        return u - self.drift_weight * (d_perp @ self._K[layer].T)

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
                self._active = True
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
            self._active = False
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