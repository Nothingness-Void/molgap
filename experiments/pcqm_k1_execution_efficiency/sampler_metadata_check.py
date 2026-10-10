"""CPU integer-only fixture timing; no dataset, model, CUDA or RML evidence."""
from __future__ import annotations

import json
import statistics
import sys
import time

from molgap.edge_state_training_core import (
    EpochPermutationBatchSampler,
    epoch_permutation_state,
)


def measure():
    import torch
    from molgap.edge_state_training_core import _epoch_permutation_metadata

    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a disposable CPU-only process")
    rows, batch_size = 500_000, 128
    full_batches = rows // batch_size
    coordinates = [(epoch, cursor) for cursor in range(128, full_batches + 1, 128)
                   for epoch in (2, 3)]

    def original():
        return [EpochPermutationBatchSampler(rows, batch_size, seed=42, epoch=epoch,
                                            start_batch=cursor).state_for(cursor)
                for epoch, cursor in coordinates]

    def cached():
        return [epoch_permutation_state(rows, batch_size, seed=42, epoch=epoch,
                                        next_batch=cursor)
                for epoch, cursor in coordinates]

    windows = []
    for window in range(4):
        _epoch_permutation_metadata.cache_clear()
        results, seconds = {}, {}
        names = ("original", "cached") if window % 2 == 0 else ("cached", "original")
        for name in names:
            tick = time.perf_counter()
            results[name] = (original if name == "original" else cached)()
            seconds[name] = time.perf_counter() - tick
        if results["original"] != results["cached"]:
            raise RuntimeError("Synthetic checkpoint sampler state changed")
        windows.append(seconds)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    medians = {name: statistics.median(w[name] for w in windows)
               for name in ("original", "cached")}
    return {
        "scope": "synthetic integer sampler metadata only; no molecular roles",
        "python": sys.version, "torch": torch.__version__, "cpu_threads": 4,
        "rows": rows, "batch_size": batch_size, "states_per_window": len(coordinates),
        "distinct_epoch_keys": 2, "windows_seconds": windows,
        "median_seconds": medians, "metadata_exact": True,
        "cuda_initialized": False,
        "training_speed_qualified": False, "rml_training_evidence": False,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
