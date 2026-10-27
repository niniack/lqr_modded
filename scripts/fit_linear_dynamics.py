"""Fit raw, last-token block dynamics for the Gemma-2-2B toxicity baseline."""

import argparse
import gc
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch


def fit_block_dynamics(activations: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Fit h[layer + 1] = A[layer] @ h[layer], without an intercept.

    Input is [examples, layer boundaries, hidden dimension]. CPU SVD least
    squares handles numerically rank-deficient activation matrices without
    forming normal equations or adding a ridge penalty.
    """
    activations = activations.detach().to(device="cpu", dtype=torch.float32)
    _, boundaries, dimension = activations.shape
    matrices = torch.empty((boundaries - 1, dimension, dimension))
    ranks = []
    for layer in range(boundaries - 1):
        result = torch.linalg.lstsq(
            activations[:, layer], activations[:, layer + 1], driver="gelsd"
        )
        # lstsq solves X @ W = Y; LQR uses column states, h_next = A @ h.
        matrices[layer].copy_(result.solution.T)
        ranks.append(int(result.rank))
        print(f"Fitted block {layer + 1}/{boundaries - 1}: rank {ranks[-1]}/{dimension}", flush=True)
    return matrices, ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096, help="Non-toxic RTP calibration prompts (default: 4096).")
    parser.add_argument("--batch-size", type=int, default=8, help="Activation collection batch size (default: 8).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New pickle path; defaults to gemma-2-2b-nontox_linear.pkl in pickle_jar.")
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("--samples and --batch-size must be positive")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from steer.data_handling import ContrastiveBuilder, PICKLE_JAR
    from steer.toxicity.tox_data_script import get_tox_prompts, load_model

    output = args.output or PICKLE_JAR / "gemma-2-2b-nontox_linear.pkl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pool = get_tox_prompts(0.0, 0.1)
    if args.samples > len(pool):
        parser.error(f"Only {len(pool)} eligible non-toxic RTP prompts are available")
    calibration_prompts = random.sample(pool, args.samples)
    model_name = "google/gemma-2-2b"
    model, tokenizer = load_model(model_name, quant=True)
    collector = ContrastiveBuilder(model, tokenizer)
    print(f"Collecting {args.samples} last-token activation trajectories", flush=True)
    activations = collector.collect_activations(
        calibration_prompts, args.samples, num_tokens=1, batch_size=args.batch_size
    )
    del collector, model
    gc.collect()
    torch.cuda.empty_cache()
    matrices, ranks = fit_block_dynamics(activations)
    artifact = {
        "A": matrices,
        "model": model_name,
        "seed": args.seed,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "training_prompts": calibration_prompts,
        "fit": {
            "method": "ordinary_least_squares",
            "intercept": False,
            "coordinates": "raw",
            "token": "last",
            "solver": "torch.linalg.lstsq:gelsd",
            "dtype": "float32",
            "rcond": None,
            "ranks": ranks,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        pickle.dump(artifact, handle)
    print(f"Saved fitted dynamics {tuple(matrices.shape)}: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
