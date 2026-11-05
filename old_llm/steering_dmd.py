"""Steering-pair DMD: per-layer operators fitted on real steering deviations.

Data come from SteeringDiagnostics, which replays each steered generation without
control on the same consumed tokens. For block l, with push u_l added to its
last-token output:

    dx_l     = x_l(steered) - x_l(replay)          block input deviation
    dx_{l+1} = x_{l+1}(steered) - x_{l+1}(replay)  block output deviation, includes u_l
    r_l      = dx_{l+1} - u_l - dx_l ~= B_l dx_l,  A_l = I + B_l

B_l is a ridge fit shrunk toward the mean-Jacobian correction Jbar_l - I with
strength c: c = 0 is plain DMD, large c returns the mean Jacobian. Only running
sums are kept. Completions are split by id: ids % 4 in {0, 1} train, 2 validation
(chooses c per layer), 3 test (reported). Deviations include KV-history effects,
so these operators are fitted to cached generation, unlike frozen-context ones.
"""

from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

__all__ = ["PairStats", "PairSink", "fit_and_evaluate"]

_SPLITS = (("train", (0, 1)), ("val", (2,)), ("test", (3,)))


class PairStats:
    def __init__(self, num_layers, hidden):
        self.xx = torch.zeros(num_layers, hidden, hidden)
        self.rx = torch.zeros(num_layers, hidden, hidden)
        self.rr = torch.zeros(num_layers, dtype=torch.float64)
        self.nn = torch.zeros(num_layers, dtype=torch.float64)
        self.count = torch.zeros(num_layers, dtype=torch.long)

    @torch.no_grad()
    def add(self, layer, dx, r, dx_next):
        """dx, r, dx_next: (N, H) float32 rows on any device."""
        self.xx[layer] += (dx.T @ dx).cpu()
        self.rx[layer] += (r.T @ dx).cpu()
        self.rr[layer] += r.double().square().sum().cpu()
        self.nn[layer] += dx_next.double().square().sum().cpu()
        self.count[layer] += dx.shape[0]

    def layer(self, layer, device):
        """(xx, rx, rr, nn, count) for one layer, float64 on `device`."""
        return (self.xx[layer].to(device, torch.float64), self.rx[layer].to(device, torch.float64),
                float(self.rr[layer]), float(self.nn[layer]), int(self.count[layer]))


class PairSink:
    """Receives per-layer deviations from SteeringDiagnostics and routes them to splits."""

    def __init__(self, num_layers, hidden):
        self.next_id = 0
        for name, _ in _SPLITS:
            setattr(self, name, PairStats(num_layers, hidden))

    def reserve(self, batch_size):
        ids = torch.arange(self.next_id, self.next_id + batch_size)
        self.next_id += batch_size
        return ids

    @torch.no_grad()
    def add(self, layer, delta, push, delta_next, valid, ids):
        """delta, push, delta_next: (steps, batch, H); valid: (steps, batch) bool; ids: (batch,)."""
        device = delta.device
        keep = valid.to(device) & (delta.norm(dim=-1) > 0)
        residual = delta_next - push - delta
        split = ids.to(device) % 4
        for name, members in _SPLITS:
            chosen = torch.isin(split, torch.tensor(members, device=device))
            mask = keep & chosen[None, :]
            if mask.any():
                getattr(self, name).add(layer, delta[mask], residual[mask], delta_next[mask])

    def counts(self):
        return {name: getattr(self, name).count.tolist() for name, _ in _SPLITS}

    def save(self, path):
        torch.save({"next_id": self.next_id,
                    **{name: vars(getattr(self, name)) for name, _ in _SPLITS}}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        sink = cls.__new__(cls)
        sink.next_id = data["next_id"]
        for name, _ in _SPLITS:
            stats = PairStats.__new__(PairStats)
            stats.__dict__.update(data[name])
            setattr(sink, name, stats)
        return sink


def _solve(sums, prior, c, rho):
    """Ridge solution of r ~= B dx shrunk toward `prior`; c, rho are relative to
    the mean eigenvalue of sum(dx dx^T), so they are unitless."""
    xx, rx = sums[:2]
    lam = (c + rho) * xx.trace() / xx.shape[-1]
    lhs = xx + lam * torch.eye(xx.shape[-1], dtype=xx.dtype, device=xx.device)
    return torch.linalg.solve(lhs, (rx + lam * prior).T).T


def _error(sums, B):
    """sqrt(sum |r - B dx|^2 / sum |dx_next|^2), the diagnostics' pooled model error."""
    xx, rx, rr, nn, count = sums
    if count == 0:
        return float("nan")
    sse = rr - 2 * (B * rx).sum() + ((B @ xx) * B).sum()
    return math.sqrt(max(float(sse), 0.0) / nn)


@torch.no_grad()
def fit_and_evaluate(sink, jbar=None, c_values=(0.0, 0.01, 0.1, 1.0, 10.0), rho=1e-4,
                     extra=None, device="cpu"):
    """Fit on train, choose c per layer on validation, report on test.

    extra: optional {name: (L, H, H) A stack} scored on the same test sums.
    Returns {"A": (L, H, H) CPU float32 operators, "rows": per-layer test errors}.
    Layers without training data (layer 0, where dx is zero) keep the prior.
    """
    num_layers, hidden = sink.train.xx.shape[:2]
    eye = torch.eye(hidden, dtype=torch.float64, device=device)
    operators = torch.empty(num_layers, hidden, hidden)
    rows = []
    for layer in tqdm(range(num_layers), desc="Steering-pair DMD", unit="layer"):
        prior = torch.zeros_like(eye) if jbar is None else jbar[layer].to(device, torch.float64) - eye
        train = sink.train.layer(layer, device)
        if train[-1] == 0 or not train[0].trace() > 0:
            operators[layer] = (eye + prior).float().cpu()
            continue
        val, test = sink.val.layer(layer, device), sink.test.layer(layer, device)
        fits = {c: _solve(train, prior, c, rho) for c in c_values}
        val_errors = {c: _error(val, B) for c, B in fits.items()}
        finite = {c: e for c, e in val_errors.items() if math.isfinite(e)}
        best = min(finite, key=finite.get) if finite else max(c_values)
        operators[layer] = (eye + fits[best]).float().cpu()
        row = {"layer": layer, "n_train": train[-1], "n_val": val[-1], "n_test": test[-1],
               "best_c": best, "identity": _error(test, torch.zeros_like(eye))}
        if jbar is not None:
            row["mean_j"] = _error(test, prior)
        for c, B in fits.items():
            row[f"test_c{c:g}"] = _error(test, B)
        row["steering_dmd"] = row[f"test_c{best:g}"]
        for name, stack in (extra or {}).items():
            row[name] = _error(test, stack[layer].to(device, torch.float64) - eye)
        rows.append(row)
        del fits, train, val, test
    return {"A": operators, "rows": rows}
