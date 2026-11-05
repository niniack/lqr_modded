import gc

import torch as th

def linearize(transition_functions, T, m, X_nom):
    """
    Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

    Args:
        transition_functions: Dynamics functions f_t(x, u) -> x_next
        X_nom: nominal states (T+1, k, n)
        U_nom: nominal controls (T, m)

    Returns:
        A: linearized A matrices (T, n, n)
        B: linearized B matrices (T, n, m)
    """
    U_nom = th.zeros([T, m], device=X_nom.device)
    n = X_nom.shape[-1]
    # print(X_nom.shape)

    A = th.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)
    # B = th.zeros((T, n, m), dtype=X_nom.dtype, device=X_nom.device)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)

        def f_last(x, u):
            return transition_functions[t](x,u)[..., -1, :]

        # Compute Jacobians:
        Jx = th.autograd.functional.jacobian(lambda x_: f_last(x_, u), x, create_graph=False, vectorize=True)   # shape: [n, *x.shape]
        # Ju = th.autograd.functional.jacobian(lambda u_: f_last(x, u_), u, create_graph=False, vectorize=True)   # shape: [n, *u.shape]
        A[t] = Jx[...,-1,:]
        # B[t] = Ju

    return A

def linearize_jvp_streamed_gpu(transition_functions, T, m, X_nom, A_out):
    """
    Streamed forward-mode Jacobian computation (column-by-column),
    writing directly into a preallocated GPU tensor.

    Args:
        transition_functions: list of dynamics functions
        T: time horizon
        m: control dim
        X_nom: (T+1, ..., n)
        A_out: (T, n, n) PREALLOCATED on GPU

    Returns:
        None (writes into A_out in-place)
    """
    device = X_nom.device
    dtype = X_nom.dtype
    n = X_nom.shape[-1]

    U_nom = th.zeros((T, m), device=device, dtype=dtype)

    print_curr_mem("before internal loop")

    for t in range(T):
        x = X_nom[t].detach()
        u = U_nom[t].detach()

        def f_last(x_):
            return transition_functions[t](x_, u)[..., -1, :]  # (..., n)

        for j in range(n):
            v = th.zeros_like(x)
            v[..., j] = 1.0

            # JVP: returns (f(x), J @ v)
            _, jvp_out = th.autograd.functional.jvp(
                f_last,
                (x,),
                (v,),
                create_graph=False,
                strict=False,
            )

            A_out[t, :, j] = jvp_out

            del v, jvp_out

        del x, u

        print_curr_mem("in internal loop")


def print_curr_mem(msg):
    print("======================================================================\n======================================================================")
    print(msg)
    gc.collect()
    if th.cuda.is_available():
        device_id = th.cuda.current_device()

        # Print allocated memory (currently used by tensors)
        print(f"th.cuda.memory_allocated: {th.cuda.memory_allocated(device_id)/1024**3:.3f}GB")

        # Print reserved memory (allocated by PyTorch's internal memory manager, including cached free blocks)
        print(f"th.cuda.memory_reserved: {th.cuda.memory_reserved(device_id)/1024**3:.3f}GB")

        # Print reserved memory (allocated by PyTorch's internal memory manager, including cached free blocks)
        print(f"th.cuda.max_memory_allocated: {th.cuda.max_memory_allocated(device_id)/1024**3:.3f}GB")

        # Print peak memory usage during the current process lifetime
        print(f"th.cuda.max_memory_reserved: {th.cuda.max_memory_reserved(device_id)/1024**3:.3f}GB")

        # Optional: Clear the memory cache (can make `nvidia-smi` report lower usage, but doesn't affect PyTorch's ability to allocate new tensors)
        th.cuda.empty_cache()
    else:
        print("CUDA not available")
    print("======================================================================\n======================================================================")
