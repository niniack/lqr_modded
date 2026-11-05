from __future__ import annotations

import csv
from pathlib import Path

from .steering import LQRSteering


def run_setpoint_trials(
    model,
    tokenizer,
    A,
    contrastive_vecs,
    lambdas,
    output_path: str | Path,
    num_trials: int = 1,
    max_new_tokens: int = 64,
    batch_size: int = 1,
    prompt: str = "Once upon a time",
    q: float = 0.1,
    r: float = 1.0,
    qf: float = 0.1,
    do_sample: bool = False,
    temp: float = 1.0,
    preserve_mem: bool = True,
    *,
    steer_last_n_layers: int | None = None,
) -> Path:
    """Save a lambda sweep, optionally steering only the final N decoder layers."""
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    lambda_values = list(lambdas)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controller = LQRSteering(
        model,
        tokenizer,
        A,
        contrastive_vecs,
        q=q,
        r=r,
        qf=qf,
        preserve_mem=preserve_mem,
        steer_last_n_layers=steer_last_n_layers,
    )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["lambda", "steered"])
        writer.writeheader()
        for lambda_value in lambda_values:
            for start in range(0, num_trials, batch_size):
                prompts = [prompt] * min(batch_size, num_trials - start)
                steered_outputs = controller.track_setpoint(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    lmbda=lambda_value,
                    do_sample=do_sample,
                    temp=temp,
                )
                writer.writerows(
                    {"lambda": lambda_value, "steered": output}
                    for output in steered_outputs
                )

    return output_path
