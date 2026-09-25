"""Measure local GPTrans V4 train-step throughput on train-role graphs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1 or args.eval_steps < 1:
        parser.error("warmup, steps, and eval-steps must be positive")

    import torch
    from molgap import pcqm_gptrans_v4 as v4
    from molgap.training_reproducibility import configure_fp32_determinism
    from molgap.v4_runtime import make_adamw_compat

    manifest_path = args.cache / "manifest.json"
    manifest_sha256 = file_sha256(manifest_path)
    expected_manifest_sha256 = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("Cache manifest SHA256 does not match frozen preflight input")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["geometry_shards"][:2]
    if [(item["role"], item["rows"]) for item in entries] != [("train", 50000)] * 2:
        raise RuntimeError("First two shards are not 50K train-role shards")
    paths = tuple(args.cache / item["file"] for item in entries)
    for path, entry in zip(paths, entries, strict=True):
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Train shard SHA256 mismatch: {path}")

    configure_fp32_determinism(v4.SEED)
    v4.LOADER_WORKERS = 0  # Windows spawn cannot pickle the local PackedGraphs class.
    graphs, shards = v4._load_datasets(paths)
    if len(graphs) != v4.TRAIN_ROWS:
        raise RuntimeError("Train-role row count changed")
    mean_value, std_value = v4._target_stats(shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    model = v4._make_model().to("cuda")
    optimizer = make_adamw_compat(
        model.parameters(), lr=v4.LEARNING_RATE, weight_decay=v4.WEIGHT_DECAY,
        fused=False, foreach=False,
    )
    ema = v4.ExponentialMovingAverage(model)
    loader = iter(v4._training_loader(graphs, 0))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.warmup):
        batch = next(loader).to("cuda", non_blocking=True)
        if int(batch.num_graphs) != v4.PHYSICAL_BATCH:
            raise RuntimeError("Physical batch changed")
        v4._optimizer_step(model, optimizer, ema, batch, mean, std, check_finite=True)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(args.steps):
        batch = next(loader).to("cuda", non_blocking=True)
        v4._optimizer_step(model, optimizer, ema, batch, mean, std, check_finite=False)
    torch.cuda.synchronize()
    measured_seconds = time.perf_counter() - started
    step_seconds = measured_seconds / args.steps
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            batch = next(loader).to("cuda", non_blocking=True)
            v4._forward(model, batch)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - started
    eval_step_seconds = eval_seconds / args.eval_steps
    total_memory = torch.cuda.get_device_properties(0).total_memory
    result = {
        "format": "molgap-local-gptrans-v4-train-role-throughput-v1",
        "scope": "train-role short preflight; not an end-to-end training result",
        "cache_manifest_sha256": manifest_sha256,
        "train_shard_sha256": [item["sha256"] for item in entries],
        "train_rows": len(graphs),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_reserve_fraction": 1 - torch.cuda.max_memory_reserved() / total_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "precision": "fp32",
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "physical_batch": v4.PHYSICAL_BATCH,
        "loader_workers": v4.LOADER_WORKERS,
        "warmup_steps": args.warmup,
        "warmup_seconds": warmup_seconds,
        "measured_steps": args.steps,
        "measured_seconds": measured_seconds,
        "step_seconds": step_seconds,
        "eval_steps_on_train_role": args.eval_steps,
        "eval_seconds": eval_seconds,
        "eval_step_seconds": eval_step_seconds,
        "estimated_60_epoch_train_hours_only": step_seconds * v4.BATCHES_PER_EPOCH * v4.EPOCHS / 3600,
        "estimated_60_epoch_train_plus_50k_eval_hours": (
            step_seconds * v4.BATCHES_PER_EPOCH * v4.EPOCHS
            + eval_step_seconds * ((v4.DEVELOPMENT_ROWS + v4.PHYSICAL_BATCH - 1) // v4.PHYSICAL_BATCH) * v4.EPOCHS
        ) / 3600,
        "unmeasured_overhead": ["checkpointing", "epoch startup", "full-run thermal behavior", "role-specific evaluation differences"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
