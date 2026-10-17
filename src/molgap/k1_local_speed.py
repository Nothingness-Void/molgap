"""Read-only historical speed acceptance helpers; the live runner is retired."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import k1_screen_training as owner
from .training_reproducibility import sha256_file
from .training_runtime import load_frozen_initial_state, validate_standard_source_bundle

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
    """Historical dual-backend runs require their immutable source package."""
    raise ValueError("Retired K1 speed runner: use frozen source for replay; registered trainers for new work")


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
