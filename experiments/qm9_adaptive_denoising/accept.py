"""No-inference acceptance for the adaptive local-denoising screen."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from molgap.qm9_adaptive_denoising import (
    BATCH_SIZE,
    MIN_GAIN_VS_FIXED_EV,
    MIN_GAIN_VS_SCRATCH_EV,
    SPLIT_FINGERPRINT,
    TOTAL_ENCODER_EPOCHS,
    sha256_file,
)
from molgap.screen_policy import validate_paired_screen_contract


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_stage(
    stage_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
    arm: str,
    phase: str,
    epochs: int,
) -> dict:
    trace = json.loads((stage_root / "trace.json").read_text(encoding="utf-8"))[
        "epochs"
    ]
    _require(len(trace) == epochs, f"{arm}/{phase} trace length changed")
    _require(
        [row["epoch"] for row in trace] == list(range(epochs)),
        f"{arm}/{phase} epoch sequence changed",
    )
    checkpoint = torch.load(
        stage_root / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    expected = {
        "epoch": epochs - 1,
        "source_commit": source_commit,
        "cache_sha256": cache_sha256,
        "phase": phase,
        "arm": arm,
        "max_epochs": epochs,
        "batch_size": BATCH_SIZE,
        "seed": 42,
        "precision_verified": True,
        "autocast_enabled": False,
    }
    for key, value in expected.items():
        _require(
            checkpoint.get(key) == value,
            f"{arm}/{phase} checkpoint changed for {key}",
        )
    for key in ("optimizer", "scheduler", "rng", "model"):
        _require(key in checkpoint, f"{arm}/{phase} checkpoint lacks {key}")
    for value in checkpoint["model"].values():
        if torch.is_tensor(value) and value.is_floating_point():
            _require(
                value.dtype == torch.float32,
                f"{arm}/{phase} model checkpoint is not FP32",
            )
    return checkpoint


def accept(root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "format": "molgap-qm9-adaptive-denoising-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "required_gain_vs_scratch_eV": MIN_GAIN_VS_SCRATCH_EV,
        "required_gain_vs_fixed_eV": MIN_GAIN_VS_FIXED_EV,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        _require(metrics.get(key) == value, f"Metrics contract changed for {key}")
        _require(
            completion.get(key) == value, f"Completion contract changed for {key}"
        )
    names = ("scratch40", "fixed10_gap30", "adaptive10_gap30")
    results = metrics["results"]
    _require(set(results) == set(names), "Unexpected denoising arm set")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in names]
    )
    _require(comparability == metrics["comparability"], "Comparability changed")

    model_hashes = {results[name]["initial_model_sha256"] for name in names}
    _require(len(model_hashes) == 1, "Initial downstream model hashes differ")
    _require(
        all(results[name]["encoder_epochs_completed"] == TOTAL_ENCODER_EPOCHS for name in names),
        "Encoder exposure differs",
    )
    _require(
        all(results[name]["contract"]["physical_batch_per_device"] == BATCH_SIZE for name in names),
        "Physical batch differs",
    )
    _require(
        results["fixed10_gap30"]["pretraining"]["denoising_head_initial_sha256"]
        == results["adaptive10_gap30"]["pretraining"]["denoising_head_initial_sha256"],
        "Denoising head initialization differs",
    )

    recomputed = {}
    for name in names:
        _require(results[name]["source_commit"] == source_commit, f"{name} source changed")
        _require(
            results[name]["cache_aggregate_sha256"] == cache_sha256,
            f"{name} cache changed",
        )
        _require(results[name]["precision_verified"] is True, f"{name} FP32 unverified")
        _require(results[name]["autocast_enabled"] is False, f"{name} autocast changed")
        training = results[name]["training"]
        gap_epochs = 40 if name == "scratch40" else 30
        _validate_stage(
            root / name / "gap",
            source_commit=source_commit,
            cache_sha256=cache_sha256,
            arm=name,
            phase="direct_gap",
            epochs=gap_epochs,
        )
        payload_path = root / name / "gap" / "best_validation_payload.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        value = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        _require(math.isfinite(value), f"Non-finite validation metric for {name}")
        _require(
            abs(value - training["validation_gap_mae_eV"]) <= 1e-9,
            f"Validation metric changed for {name}",
        )
        _require(
            sha256_file(payload_path) == training["payload_sha256"],
            f"Validation payload hash changed for {name}",
        )
        for field in (
            "parameter_count",
            "inference_parameter_count",
        ):
            _require(int(results[name][field]) > 0, f"Invalid {field} for {name}")
        for field in ("throughput_graphs_per_s", "peak_memory_mib"):
            _require(
                math.isfinite(float(training[field])) and float(training[field]) > 0,
                f"Invalid {field} for {name}",
            )
        recomputed[name] = value

    for name in ("fixed10_gap30", "adaptive10_gap30"):
        pretrain = results[name]["pretraining"]
        checkpoint = _validate_stage(
            root / name / "pretrain",
            source_commit=source_commit,
            cache_sha256=cache_sha256,
            arm=name,
            phase=(
                "adaptive_denoising"
                if name == "adaptive10_gap30"
                else "fixed_denoising"
            ),
            epochs=10,
        )
        rng_contract = checkpoint.get("rng_contract")
        _require(
            rng_contract
            == {
                "model_seed": 42,
                "auxiliary_head_seed": 43,
                "adaptive_generator_seed": 44,
                "corruption_seed": 45,
                "training_stream_seed": 142,
                "loader_seed": 42,
            },
            f"{name} RNG contract changed",
        )
        _require("head" in checkpoint and "generator" in checkpoint, f"{name} auxiliary state missing")
        if name == "fixed10_gap30":
            _require(checkpoint["generator"] is None, "Fixed arm gained a generator")
        else:
            _require(isinstance(checkpoint["generator"], dict), "Adaptive generator missing")
        for field in (
            "pretrain_loss",
            "sigma_mean",
            "sigma_std",
            "throughput_graphs_per_s",
            "peak_memory_mib",
        ):
            _require(math.isfinite(float(pretrain[field])), f"Invalid {field} for {name}")

    for relative, expected in completion["artifact_sha256"].items():
        _require(
            sha256_file(root / relative) == expected,
            f"Artifact hash changed: {relative}",
        )
    gain_scratch = recomputed["scratch40"] - recomputed["adaptive10_gap30"]
    gain_fixed = recomputed["fixed10_gap30"] - recomputed["adaptive10_gap30"]
    nominated = (
        gain_scratch >= MIN_GAIN_VS_SCRATCH_EV
        and gain_fixed >= MIN_GAIN_VS_FIXED_EV
    )
    _require(
        nominated == metrics["pcqm100k_transfer_nominated"],
        "Stored transfer decision changed",
    )
    return {
        "format": "molgap-qm9-adaptive-denoising-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "recomputed_validation_gap_mae_eV": recomputed,
        "adaptive_gain_vs_scratch_eV": gain_scratch,
        "adaptive_gain_vs_fixed_eV": gain_fixed,
        "pcqm100k_transfer_nominated": nominated,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(
        args.root,
        source_commit=args.source_commit,
        cache_sha256=args.cache_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
