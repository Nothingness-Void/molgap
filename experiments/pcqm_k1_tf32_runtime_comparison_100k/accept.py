"""Independently accept retrieved FP32/TF32 artifacts without model inference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from molgap.pcqm_k1_tf32_comparison import (
    ARMS,
    BATCH_SIZE,
    DEVELOPMENT_ROWS,
    EPOCHS,
    FIXED_MANIFEST_SHA256,
    FORMAT,
    PARAMETERS,
    ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE,
    STEPS_PER_EPOCH,
)
from molgap.training_reproducibility import sha256_file


def accept(output: Path) -> dict:
    import torch

    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    if (
        comparison.get("format") != FORMAT
        or comparison.get("complete") is not True
        or comparison.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or set(comparison.get("arms", {})) != set(ARMS)
    ):
        raise RuntimeError("Comparison manifest is incomplete")
    initial_shas = set()
    aligned = []
    accepted_arms = {}
    for arm in ARMS:
        directory = output / arm
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        runtime = json.loads((directory / "runtime_certificate.json").read_text(encoding="utf-8"))
        trace = json.loads((directory / "trace.json").read_text(encoding="utf-8"))["epochs"]
        if (
            metrics.get("format") != FORMAT
            or metrics.get("complete") is not True
            or metrics.get("arm") != arm
            or metrics.get("parameter_count") != PARAMETERS
            or metrics.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
            or metrics.get("physical_batch") != BATCH_SIZE
            or metrics.get("sample_presentations") != SAMPLE_EXPOSURE
            or metrics.get("optimizer_steps") != EPOCHS * STEPS_PER_EPOCH
            or metrics.get("epochs_completed") != EPOCHS
            or runtime.get("calibration_checks_passed") is not True
            or runtime.get("precision", {}).get("tf32_enabled") != (arm == "tf32_matmul")
            or any(metrics.get(role) is not False for role in (
                "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"
            ))
        ):
            raise RuntimeError(f"Invalid terminal identity: {arm}")
        for name, digest in metrics["artifact_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise RuntimeError(f"Artifact SHA-256 mismatch: {arm}/{name}")
        if len(trace) != EPOCHS or any(
            row["epoch"] != epoch
            or row["optimizer_steps"] != (epoch + 1) * STEPS_PER_EPOCH
            or row["sample_presentations"] != (epoch + 1) * ROWS_PER_EPOCH
            or not all(math.isfinite(float(row[field])) for field in (
                "train_normalized_mae", "development_gap_mae_eV", "training_seconds", "validation_seconds"
            ))
            for epoch, row in enumerate(trace)
        ):
            raise RuntimeError(f"Trace is incomplete or nonfinite: {arm}")
        if abs(min(row["development_gap_mae_eV"] for row in trace) - metrics["best_development_gap_mae_eV"]) > 1e-7:
            raise RuntimeError(f"Best metric/trace mismatch: {arm}")
        payload = torch.load(directory / "best_development_payload.pt", map_location="cpu", weights_only=False)
        source_idx = payload["source_idx"].view(-1).long()
        target = payload["target_eV"].view(-1).float()
        prediction = payload["prediction_eV"].view(-1).float()
        if (
            len(source_idx) != DEVELOPMENT_ROWS
            or not torch.equal(source_idx, torch.arange(100_000, 150_000))
            or not bool(torch.isfinite(target).all())
            or not bool(torch.isfinite(prediction).all())
        ):
            raise RuntimeError(f"Prediction role or values changed: {arm}")
        recomputed = float((prediction - target).abs().mean().item())
        if abs(recomputed - metrics["best_development_gap_mae_eV"]) > 2e-6:
            raise RuntimeError(f"Saved predictions do not reproduce MAE: {arm}")
        aligned.append((source_idx, target))
        initial_shas.add(metrics["initial_model_sha256"])
        accepted_arms[arm] = {
            "best_development_gap_mae_eV": recomputed,
            "best_epoch": metrics["best_epoch"],
            "training_seconds": metrics["training_seconds"],
            "training_graphs_per_second": metrics["training_graphs_per_second"],
            "terminal_metrics_sha256": sha256_file(directory / "metrics.json"),
        }
    if len(initial_shas) != 1 or any(
        not torch.equal(aligned[0][index], aligned[1][index]) for index in (0, 1)
    ):
        raise RuntimeError("Arms are not initialized/aligned on the same role")
    if (
        abs(
            comparison["tf32_minus_fp32_mae_eV"]
            - (accepted_arms["tf32_matmul"]["best_development_gap_mae_eV"]
               - accepted_arms["fp32"]["best_development_gap_mae_eV"])
        ) > 2e-6
        or abs(
            comparison["training_speedup_fp32_over_tf32"]
            - accepted_arms["fp32"]["training_seconds"]
              / accepted_arms["tf32_matmul"]["training_seconds"]
        ) > 1e-6
    ):
        raise RuntimeError("Comparison summary does not recompute")
    return {
        "format": "molgap-k1-tf32-no-inference-acceptance-v1",
        "accepted": True,
        "model_inference_executed": False,
        "source_commit": comparison["source_commit"],
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "arms": accepted_arms,
        "initial_model_sha256": next(iter(initial_shas)),
        "tf32_minus_fp32_mae_eV": comparison["tf32_minus_fp32_mae_eV"],
        "training_speedup_fp32_over_tf32": comparison["training_speedup_fp32_over_tf32"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.output)
    args.acceptance.parent.mkdir(parents=True, exist_ok=True)
    if args.acceptance.exists():
        raise FileExistsError(args.acceptance)
    args.acceptance.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
