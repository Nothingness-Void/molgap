"""CPU synthetic log sizing with existing atomic IO; never a training trace."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch

from molgap import training_reproducibility as owner


ROOT = Path(__file__).resolve().parents[2]


def measure():
    tmp = ROOT / ".tmp"
    tmp.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="k1-json-fixture-", dir=tmp)).resolve()
    if scratch.parent != tmp.resolve() or scratch.is_symlink():
        raise RuntimeError("Scratch is not confined to the workspace temporary directory")
    started = time.perf_counter()
    results = []
    original_dumps, original_write = json.dumps, Path.write_text
    paths = [scratch / "pretty.json", scratch / "compact.json"]
    try:
        for size in (0, 8192, 65536, 234360):
            # Invented integers/scalars for formatting only, not a learning record.
            rows = [dict(step=i + 1, epoch=i // 3906, next_batch=i % 3906 + 1,
                         samples=(i + 1) * 128, loss=0.5, seconds=0.125,
                         allocation_seconds=(i + 1) * 0.125) for i in range(size)]
            payload = {name: dict(epochs=[], steps=rows) for name in ("reference", "ema999")}
            windows = []
            for window in range(4):
                pair = {}
                names = ("pretty", "compact") if window % 2 == 0 else ("compact", "pretty")
                for name in names:
                    if time.perf_counter() - started > 120:
                        raise TimeoutError("Synthetic publication check exceeded120s")
                    times = {}

                    def timed_dumps(*args, **kwargs):
                        tick = time.perf_counter()
                        value = original_dumps(*args, **kwargs)
                        times["encode_seconds"] = time.perf_counter() - tick
                        return value

                    def timed_write(path, *args, **kwargs):
                        tick = time.perf_counter()
                        value = original_write(path, *args, **kwargs)
                        times["write_seconds"] = time.perf_counter() - tick
                        return value

                    path = scratch / f"{name}.json"
                    with patch.object(owner.json, "dumps", timed_dumps), patch.object(Path, "write_text", timed_write):
                        tick = time.perf_counter()
                        owner.atomic_json(path, payload, compact=name == "compact")
                        times["atomic_publication_seconds"] = time.perf_counter() - tick
                    tick = time.perf_counter()
                    digest = owner.sha256_file(path)
                    times["hash_seconds"] = time.perf_counter() - tick
                    times["publication_hash_seconds"] = times["atomic_publication_seconds"] + times["hash_seconds"]
                    pair[name] = dict(times, file_bytes=path.stat().st_size, sha256=digest)
                windows.append(pair)
            # Check content outside the timing, including field order and scalar types.
            for path in paths:
                with path.open(encoding="utf-8") as handle:
                    if json.load(handle) != payload:
                        raise RuntimeError("Synthetic log values changed")
            results.append({
                "synthetic_rows_per_arm": size, "windows": windows,
                "median_seconds": {
                    name: {key: statistics.median(w[name][key] for w in windows)
                           for key in ("encode_seconds", "write_seconds", "hash_seconds",
                                       "atomic_publication_seconds", "publication_hash_seconds")}
                    for name in ("pretty", "compact")},
                "logical_payload_equal": True,
            })
        return dict(scope="synthetic CPU JSON only; no dataset/checkpoint/model/CUDA",
                    source_default_unchanged=True, synthetic_not_rml_training_evidence=True,
                    native_training_saving_qualified=False, measured=results,
                    fixture_wall_seconds=time.perf_counter() - started,
                    storage_limits="Local page-cache writes; no fsync, remote IO or native allocation qualification")
    finally:
        # Remove only the known generated files under the verified scratch directory.
        for path in paths:
            for owned in (path, path.with_name(f".{path.name}.tmp")):
                if owned.exists():
                    owned.unlink()
        scratch.rmdir()


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
