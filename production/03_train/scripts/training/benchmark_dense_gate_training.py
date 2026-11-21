"""Benchmark the repaired-2M dense gate on synthetic expert predictions."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch


def traced_method(original, name, calls):
    def traced(self, *args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_globals.get("__name__") == "molgap.multi2d_router_fusion":
            calls[f"{caller.f_code.co_name}:{caller.f_lineno}:{name}"] += 1
        return original(self, *args, **kwargs)
    return traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--count-syncs", action="store_true")
    parser.add_argument("--router", action="store_true")
    args = parser.parse_args()
    if args.source is None:
        from molgap.constants import REPO_ROOT

        args.source = REPO_ROOT / "src"
    else:
        sys.path.insert(0, str(args.source.resolve()))
    from molgap.multi2d_router_fusion import GateTrainingConfig, fit_dense_soft_gate, fit_predispatch_router

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(42)
    targets = rng.normal(size=(args.rows, 3)).astype(np.float32)
    noise = rng.normal(size=(args.rows, 3, 3)).astype(np.float32)
    predictions = targets[:, None, :] + noise * np.float32(0.2)
    split = args.rows * 4 // 5
    train = np.arange(split)
    validation = np.arange(split, args.rows)
    config = GateTrainingConfig(epochs=args.epochs, patience=args.epochs)
    if args.router:
        descriptors = rng.normal(size=(args.rows, 5)).astype(np.float32)

    def fit():
        if args.router:
            return fit_predispatch_router(predictions, targets, descriptors, train, validation, config=config)
        return fit_dense_soft_gate(predictions, targets, train, validation, config=config)

    times = []
    for _ in range(args.repeats + 1):
        start = perf_counter()
        model, report = fit()
        times.append(perf_counter() - start)
    calls = Counter()
    if args.count_syncs:
        with ExitStack() as stack:
            for name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__", "__int__", "nonzero"):
                method = traced_method(getattr(torch.Tensor, name), name, calls)
                stack.enter_context(patch.object(torch.Tensor, name, method))
            fit()
    state_hash = sha256()
    for tensor in model.state_dict().values():
        state_hash.update(tensor.detach().numpy().tobytes())
    report_hash = sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    print(json.dumps({
        **vars(args), "torch": torch.__version__, "batch_size": config.batch_size,
        "median_seconds": round(median(times[1:]), 3),
        "repeat_seconds": [round(value, 3) for value in times[1:]],
        "best_validation_loss": report["best_validation_utility_loss" if args.router else "best_validation_mae_eV"],
        "checkpoint_sha256": state_hash.hexdigest(), "report_sha256": report_hash,
        "read_calls": calls,
    }, default=str))


if __name__ == "__main__":
    main()
