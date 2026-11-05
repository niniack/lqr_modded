"""Drift-aware A-LQR steering (cost B).

The original controller's cost Q = q I already penalizes off-v deviation, but its
error signal (beta_l - v_l^T x_l) v_l lies along v, so it never sees drift. Here the
error also includes -P_perp,l d_l, where d_l is the deviation this token's own
pushes are predicted to cause under the dynamics model:

    d_{l+1} = A_l d_l + u_l,   d = 0 at the first steered layer
    e_l     = (beta_l - v_l^T x_l) v_l - P_perp,l d_l,   u_l = K_l e_l

Cost: Q_l = q (v_l v_l^T + rho P_perp,l), Qf = qf (v_L v_L^T + rho P_perp,L), R = r I.
rho = 1 gives the original Q and gains; observe_drift=False gives the original
controller. d ignores steering of earlier tokens through the KV cache.
"""

from __future__ import annotations

import torch

from .steering import LQRSteering

__all__ = ["DriftLQRSteering", "drift_lqr_gains"]


def drift_lqr_gains(A, directions, q, r, qf, rho=1.0, device="cpu"):
    """Riccati gains for e_{l+1} = A_l e_l - u_l, u_l = K_l e_l; (L, H, H) float32 on CPU."""
    num_layers, hidden = A.shape[0], A.shape[-1]
    eye = torch.eye(hidden, dtype=torch.float64, device=device)

    def weight(scale, direction):
        v = direction.to(device, torch.float64)
        v = v / v.norm().clamp_min(1e-12)
        return scale * (rho * eye + (1 - rho) * torch.outer(v, v))

    P = weight(qf, directions[num_layers])
    K = torch.empty(num_layers, hidden, hidden)
    for layer in reversed(range(num_layers)):
        a = A[layer].to(device, torch.float64)
        gain = torch.linalg.solve(r * eye + P, P @ a)
        K[layer] = gain.float().cpu()
        P = weight(q, directions[layer]) + a.T @ P @ (a - gain)
        P = (P + P.T) / 2
    return K


class DriftLQRSteering(LQRSteering):
    """LQRSteering with cost B; reuses its track_setpoint and hook registration."""

    def __init__(self, model, tokenizer, A, contrastive_vecs, q, r, qf, rho=1.0,
                 observe_drift=True, *, steer_last_n_layers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = model.model.layers
        self.T = len(self.layers)
        self.n = model.get_input_embeddings().weight.shape[1]
        if A.shape != (self.T, self.n, self.n) or contrastive_vecs.shape != (self.T + 1, self.n):
            raise ValueError("A must be (L, H, H) and contrastive_vecs (L + 1, H).")
        steered = self.T if steer_last_n_layers is None else steer_last_n_layers
        self.steered_layer_indices = range(self.T - steered, self.T)
        self.input_device = model.get_input_embeddings().weight.device
        self.device = next(self.layers[0].parameters()).device
        self.E = contrastive_vecs.detach().to(self.device, torch.float32)
        self.K = drift_lqr_gains(A, contrastive_vecs, q, r, qf, rho, self.device)
        devices = [next(layer.parameters()).device for layer in self.layers]
        self._layer_gains = [self.K[i].to(device) for i, device in enumerate(devices)]
        self._layer_A = ([A[i].to(device, torch.float32) for i, device in enumerate(devices)]
                         if observe_drift else None)
        self.rho = rho
        self.observe_drift = observe_drift
        self.ALL_TOKENS = False
        self.hooks = []
        self._drift = None

    def hook_setpoint_tracking(self, layer_idx, module, module_inputs, module_output):
        if self.ALL_TOKENS:
            raise ValueError("DriftLQRSteering steers the last token only.")
        x = module_inputs[0][:, -1, :].float()
        v = self._layer_directions[layer_idx].to(x.device)
        error = (self._layer_betas[layer_idx].to(x.device) - x @ v)[:, None] * v
        if self.observe_drift:
            if layer_idx == self.steered_layer_indices[0]:
                drift = torch.zeros_like(x)
            else:
                drift = self._drift.to(x.device)
            error = error - (drift - (drift @ v)[:, None] * v)
        control = error @ self._layer_gains[layer_idx].T
        if self.observe_drift:
            self._drift = drift @ self._layer_A[layer_idx].T + control

        hidden = module_output[0] if isinstance(module_output, (tuple, list)) else module_output
        updated = hidden.clone()
        updated[:, -1, :] += control.to(updated.dtype)
        if isinstance(module_output, tuple):
            return (updated, *module_output[1:])
        if isinstance(module_output, list):
            return [updated, *module_output[1:]]
        return updated
