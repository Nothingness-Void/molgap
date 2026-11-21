"""Benchmark the repaired-2M dense gate on synthetic expert predictions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[4] / "src")
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from molgap.multi2d_router_fusion import GateTrainingConfig, fit_dense_soft_gate

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(42)
    targets = rng.normal(size=(args.rows, 3)).astype(np.float32)
    noise = rng.normal(size=(args.rows, 3, 3)).astype(np.float32)
    predictions = targets[:, None, :] + noise * np.float32(0.2)
    split = args.rows * 4 // 5
    train = np.arange(split)
    validation = np.arange(split, args.rows)
    config = GateTrainingConfig(epochs=args.epochs, patience=args.epochs)

    times = []
    for _ in range(args.repeats + 1):
        start = perf_counter()
        model, report = fit_dense_soft_gate(
            predictions, targets, train, validation, config=config,
        )
        times.append(perf_counter() - start)
    state_hash = sha256()
    for tensor in model.state_dict().values():
        state_hash.update(tensor.detach().numpy().tobytes())
    report_hash = sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    print(json.dumps({
        **vars(args), "torch": torch.__version__, "batch_size": config.batch_size,
        "median_seconds": round(median(times[1:]), 3),
        "repeat_seconds": [round(value, 3) for value in times[1:]],
        "best_validation_mae_eV": report["best_validation_mae_eV"],
        "checkpoint_sha256": state_hash.hexdigest(), "report_sha256": report_hash,
    }, default=str))


if __name__ == "__main__":
    main()
