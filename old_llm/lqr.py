from __future__ import annotations

import torch


def _validate_inputs(A, Q, R, S_T):
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("A must have shape (time, hidden, hidden).")
    time_steps, hidden_size, _ = A.shape
    expected_cost_shape = (time_steps, hidden_size, hidden_size)
    if Q.shape != expected_cost_shape or R.shape != expected_cost_shape:
        raise ValueError("Q and R must match A's (time, hidden, hidden) shape.")
    if S_T.shape != (hidden_size, hidden_size):
        raise ValueError("S_T must have shape (hidden, hidden).")
    return time_steps, hidden_size


def _riccati_step(A_t, Q_t, R_t, S_next):
    P = S_next + R_t
    F = S_next @ A_t
    G = Q_t + A_t.T @ S_next @ A_t
    K_t = torch.linalg.solve(P, F)
    S_t = G - F.T @ K_t
    return K_t, S_t


def time_varying_lqr_noB(A, Q, R, S_T):
    """Solve the finite-horizon Riccati recursion without an input matrix.

    Inputs are promoted to float32. Returns gains with shape
    ``(time, hidden, hidden)`` on ``A``'s device.
    """
    time_steps, hidden_size = _validate_inputs(A, Q, R, S_T)
    device = A.device
    A = A.to(device=device, dtype=torch.float32)
    Q = Q.to(device=device, dtype=torch.float32)
    R = R.to(device=device, dtype=torch.float32)
    S_T = S_T.to(device=device, dtype=torch.float32)

    S = torch.empty(
        time_steps + 1,
        hidden_size,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )
    K = torch.empty(
        time_steps,
        hidden_size,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )
    S[time_steps] = S_T

    for t in reversed(range(time_steps)):
        K[t], S[t] = _riccati_step(A[t], Q[t], R[t], S[t + 1])

    return K


def time_varying_lqr_noB_mem_efficient(A, Q, R, S_T, device=None):
    """Solve the same recursion while keeping the value matrices on CPU.

    ``A`` may remain on CPU while each time slice is transferred to ``device``
    for its solve. The returned gains reside on the compute device.
    """
    time_steps, hidden_size = _validate_inputs(A, Q, R, S_T)
    compute_device = torch.device(device) if device is not None else A.device
    Q = Q.to(device=compute_device, dtype=torch.float32)
    R = R.to(device=compute_device, dtype=torch.float32)
    S_T = S_T.to(device="cpu", dtype=torch.float32)

    S = torch.empty(
        time_steps + 1,
        hidden_size,
        hidden_size,
        dtype=torch.float32,
        device="cpu",
    )
    K = torch.empty(
        time_steps,
        hidden_size,
        hidden_size,
        dtype=torch.float32,
        device=compute_device,
    )
    S[time_steps] = S_T

    for t in reversed(range(time_steps)):
        A_t = A[t].to(device=compute_device, dtype=torch.float32)
        S_next = S[t + 1].to(device=compute_device)
        K_t, S_t = _riccati_step(A_t, Q[t], R[t], S_next)
        K[t] = K_t
        S[t] = S_t.to(device="cpu")

    return K
