"""Bounded local execution adapter; reuses the K1 owner, never evaluates dev."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from . import k1_screen_training as owner
from .k1_execution_layout import cpu_layout_context
from .k1_loader_reuse import K1LoaderReuse
from .training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest, capture_rng_state,
    configure_fp32_determinism, restore_rng_state, sha256_file,
)
from .v4_runtime import load_frozen_initial_state, make_adamw_compat, torch_load_compat, validate_standard_source_bundle

ARMS = ("reference", "fused_layout")
FORMAT = "molgap-k1-local-speed-v1"


def validate_config(config):
    if (config.get("format") != FORMAT or config.get("epochs") != 10
            or config.get("allocation_seconds") != 3600
            or config.get("schedule_epochs") != owner.EPOCHS
            or config.get("batch_size") != owner.BATCH_SIZE
            or config.get("development_access") is not False
            or config.get("precision") != "fp32-tf32-off"):
        raise ValueError("Expected the authorized 10ep/60min TRAIN-only FP32 contract")
    return config


def verify_source(root, config):
    if sha256_file(root / "SOURCE_FILES.json") != config["source_inventory_sha256"]:
        raise ValueError("Frozen inventory authority changed")
    validate_standard_source_bundle(Path(config["source_archive"]),
        config["source_package_sha256"], config["source_commit"])
    inventory = json.loads((root / "SOURCE_FILES.json").read_text())
    if inventory["source_commit"] != config["source_commit"]:
        raise ValueError("Source commit changed")
    for item in inventory["files"]:
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['path']}")


def cpu_inputs(config):
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Graph acceptance must precede accelerator initialization")
    root, manifest = owner.find_fixed_cache(Path(config["input_root"]))
    train = owner.load_roles(root, manifest, selected_roles=("train",))["train"]
    mean, std = owner._target_stats(train)
    model = owner.make_encoder("neural_atom_k1")
    initial = load_frozen_initial_state(
        model, Path(config["initial_path"]),
        expected_file_sha256=config["initial_file_sha256"],
        expected_state_sha256=config["initial_tensor_sha256"],
        expected_format=config["initial_format"],
    )
    return train, mean, std, {k: v.clone() for k, v in model.state_dict().items()}, initial


def summarize(rows):
    result = {}
    for arm in ARMS:
        all_rows = [r for r in rows if r["arm"] == arm]
        steady = [r for r in all_rows if r["batch"] >= 8]
        result[arm] = {
            "steps": len(all_rows), "samples": sum(r["samples"] for r in all_rows),
            "optimizer_step_seconds": sum(r["step_seconds"] for r in all_rows),
            "arm_pipeline_seconds": sum(r["pipeline_seconds"] for r in all_rows),
            "steady_step_median_seconds": statistics.median(r["step_seconds"] for r in steady) if steady else None,
            "steady_pipeline_median_seconds": statistics.median(r["pipeline_seconds"] for r in steady) if steady else None,
        }
    if result[ARMS[0]]["steps"] != result[ARMS[1]]["steps"]:
        raise ValueError("Unmatched pair exposure")
    ref, cand = (result[a]["arm_pipeline_seconds"] for a in ARMS)
    result["pipeline_time_reduction_fraction"] = 1 - cand / ref if ref else None
    result["quality_evaluated"] = False
    return result


def run(config_path, output, source_root, *, resume=False):
    import torch
    from torch_geometric.loader import DataLoader

    config = validate_config(json.loads(config_path.read_text()))
    verify_source(source_root, config)
    for arm in ARMS:
        path = Path(config["prospective"][arm]["path"])
        if sha256_file(path) != config["prospective"][arm]["sha256"]:
            raise ValueError("Prospective identity changed")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    if checkpoint.exists() != resume:
        raise ValueError("Existing checkpoint requires explicit resume; resume requires a checkpoint")
    config_sha = sha256_file(config_path)
    saved = torch_load_compat(checkpoint, map_location="cpu", weights_only=False) if resume else None
    if saved is not None and saved["config_sha256"] != config_sha:
        raise ValueError("Resume config changed")
    cpu_started = time.perf_counter()
    train, mean, std, initial, initial_identity = cpu_inputs(config)
    atomic_json(output / "cpu_acceptance.json", {
        "train_rows": len(train), "manifest_sha256": owner.FIXED_MANIFEST_SHA256,
        "initial": initial_identity, "target_mean": mean, "target_std": std,
        "development_decoded": False, "cuda_initialized": torch.cuda.is_initialized(),
        "wall_seconds": time.perf_counter() - cpu_started,
    })
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU acceptance unexpectedly initialized CUDA")

    # Allocation wall includes setup, qualification, worker startup and final IO.
    allocation_start = saved["allocation_started_unix"] if saved else time.time()
    remaining = config["allocation_seconds"] - (time.time() - allocation_start)
    if remaining <= 150:
        raise RuntimeError("No remaining allocation budget; no accelerator released")
    atomic_json(output / "allocation.json", {"started_unix": allocation_start, "limit_seconds": 3600})

    def timeout():
        atomic_json(output / "watchdog.json", {
            "status": "STOP_FOR_COST", "reason": "Hard wall ceiling; only last atomic checkpoint is authoritative",
        })
        os._exit(124)

    watchdog = threading.Timer(remaining, timeout)
    watchdog.daemon = True
    watchdog.start()
    pool = None
    models, optimizers, schedulers, rngs = {}, {}, {}, {}
    rows = saved["timings"] if saved else []
    epoch, offset = (saved["epoch"], saved["offset"]) if saved else (0, 0)
    loader_seconds = saved["loader_seconds"] if saved else 0.0
    checkpoint_seconds = saved["checkpoint_seconds"] if saved else 0.0
    status = "INFRASTRUCTURE_FAILURE"
    try:
        runtime = configure_fp32_determinism(owner.SEED)
        if "RTX 5060" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This release is local RTX5060 only")
        atomic_json(output / "runtime.json", build_runtime_manifest(runtime))
        for arm in ARMS:
            model = owner.make_encoder("neural_atom_k1")
            model.load_state_dict(initial, strict=True)
            models[arm] = model.to("cuda").train()
            optimizers[arm] = make_adamw_compat(model.parameters(), fused=arm != "reference",
                lr=owner.LEARNING_RATE, weight_decay=owner.WEIGHT_DECAY, foreach=False)
            schedulers[arm] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[arm], T_max=owner.EPOCHS, eta_min=1e-6)
        configure_fp32_determinism(owner.SEED)
        for arm in ARMS:
            rngs[arm] = capture_rng_state()
            if saved:
                models[arm].load_state_dict(saved["arms"][arm]["model"], strict=True)
                optimizers[arm].load_state_dict(saved["arms"][arm]["optimizer"])
                schedulers[arm].load_state_dict(saved["arms"][arm]["scheduler"])
                rngs[arm] = saved["arms"][arm]["rng"]
        generator = torch.Generator(device="cpu").manual_seed(owner.SEED)
        if saved:
            generator.set_state(saved["loader_generator"])
        loader = DataLoader(train, sampler=[], batch_size=owner.BATCH_SIZE,
            num_workers=2, persistent_workers=True, pin_memory=True,
            prefetch_factor=2, generator=generator, drop_last=True)
        pool = K1LoaderReuse(loader, deterministic_dataset=True)

        def save():
            nonlocal checkpoint_seconds
            started = time.perf_counter()
            torch.cuda.synchronize()
            atomic_torch_save(checkpoint, {
                "format": FORMAT, "config_sha256": config_sha,
                "allocation_started_unix": allocation_start, "epoch": epoch, "offset": offset,
                "timings": rows, "loader_seconds": loader_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "loader_generator": generator.get_state(),
                "arms": {arm: {"model": models[arm].state_dict(),
                    "optimizer": optimizers[arm].state_dict(),
                    "scheduler": schedulers[arm].state_dict(), "rng": rngs[arm]} for arm in ARMS},
            })
            checkpoint_seconds += time.perf_counter() - started
            atomic_json(output / "progress.json", {
                "epoch": epoch, "batch_offset": offset, "steps_per_arm": len(rows) // 2,
                "allocation_wall_seconds": time.time() - allocation_start,
                "status": "RUNNING", "quality_evaluated": False,
            })

        save()
        while epoch < config["epochs"]:
            started = time.perf_counter()
            with pool.stream(owner.epoch_order(epoch)[offset * owner.BATCH_SIZE:]) as iterator:
                loader_seconds += time.perf_counter() - started
                while offset < owner.STEPS_PER_EPOCH:
                    if time.time() - allocation_start >= config["allocation_seconds"] - 150:
                        status = "STOP_FOR_COST"
                        break
                    started = time.perf_counter()
                    cpu_batch = next(iterator)
                    loader_seconds += time.perf_counter() - started
                    order = ARMS if (epoch * owner.STEPS_PER_EPOCH + offset) % 2 == 0 else ARMS[::-1]
                    pending = []
                    for arm in order:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        restore_rng_state(rngs[arm])
                        batch = cpu_batch.clone()
                        context = cpu_layout_context(models[arm], batch, "cuda") if arm != "reference" else nullcontext()
                        with context:
                            batch = batch.to("cuda", non_blocking=True)
                            torch.cuda.synchronize()
                            step_started = time.perf_counter()
                            loss, _, samples = owner._optimizer_step(models[arm], optimizers[arm], batch, mean, std)
                            torch.cuda.synchronize()
                            step_seconds = time.perf_counter() - step_started
                            if not bool(torch.isfinite(loss).item()):
                                raise RuntimeError(f"Nonfinite loss: {arm}")
                        rngs[arm] = capture_rng_state()
                        pending.append({"arm": arm, "epoch": epoch, "batch": offset,
                            "samples": samples, "step_seconds": step_seconds,
                            "pipeline_seconds": time.perf_counter() - started,
                            "learning_rate": optimizers[arm].param_groups[0]["lr"]})
                        del batch, loss
                    rows.extend(pending)
                    offset += 1
                    if offset % 128 == 0:
                        save()
                if status == "STOP_FOR_COST":
                    break
            if status == "STOP_FOR_COST":
                break
            for scheduler in schedulers.values():
                scheduler.step()
            epoch, offset = epoch + 1, 0
            save()
            print(json.dumps({"completed_epochs": epoch, **summarize(rows)}), flush=True)
        status = "COMPLETE_SPEED_PREFIX" if epoch == 10 else status
        save()
        summary = {"status": status, "completed_epochs": epoch, "batch_offset": offset,
            "config_sha256": config_sha, "source_commit": config["source_commit"],
            "allocation_wall_seconds": time.time() - allocation_start,
            "loader_seconds_shared": loader_seconds, "checkpoint_seconds_shared": checkpoint_seconds,
            "checkpoint_sha256": sha256_file(checkpoint), "results": summarize(rows)}
        atomic_json(output / "timings.json", {"rows": rows})
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        atomic_json(output / "failure.json", {"type": type(error).__name__, "error": str(error),
            "last_atomic_checkpoint": str(checkpoint) if checkpoint.exists() else None,
            "partial_in_memory_state_not_saved": True})
        raise
    finally:
        try:
            if pool is not None:
                pool.close()
            atomic_json(output / "allocation_terminal.json", {
                "status": status, "started_unix": allocation_start,
                "finished_unix": time.time(), "wall_seconds": time.time() - allocation_start,
                "includes_owned_worker_cleanup": True,
                "process_exit_release_not_observed_here": True,
            })
        finally:
            watchdog.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.source_root, resume=args.resume)


if __name__ == "__main__":
    main()
