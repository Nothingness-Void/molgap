"""Mechanical acceptance for a completed GPTrans-T 100K V4 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molgap.pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH,
    DEVELOPMENT_ROWS,
    EPOCHS,
    MANIFEST_SHA256,
    PHYSICAL_BATCH,
    RUN_FORMAT,
    SAMPLE_PRESENTATIONS,
)
from molgap.screen_policy import validate_runtime_certificate
from molgap.training_reproducibility import atomic_json, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def accept(output: Path, acceptance_path: Path) -> dict:
    output = output.resolve()
    completion_path = output / "completion_manifest.json"
    reference_path = output / "frozen_reference.json"
    best_path = output / "best_model.pt"
    predictions_path = output / "development_predictions.pt"
    preflight_path = output.parent / "preflight" / "preflight.json"
    trace_path = output / "trace.json"
    for path in (completion_path, reference_path, best_path, predictions_path, preflight_path, trace_path):
        require(path.is_file(), f"missing artifact: {path}")

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["rows"]
    predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)

    require(completion.get("format") == RUN_FORMAT and completion.get("complete") is True, "completion")
    require(completion.get("epochs") == EPOCHS, "epoch count")
    require(completion.get("optimizer_steps") == BATCHES_PER_EPOCH * EPOCHS, "optimizer steps")
    require(completion.get("sample_presentations") == SAMPLE_PRESENTATIONS, "sample exposure")
    require(completion.get("manifest_sha256") == MANIFEST_SHA256, "manifest identity")
    require(len(trace) == EPOCHS, "trace length")
    require([int(row["epoch"]) for row in trace] == list(range(EPOCHS)), "trace order")
    require(all(int(row["sample_presentations"]) == BATCHES_PER_EPOCH * PHYSICAL_BATCH for row in trace), "epoch exposure")

    prediction = predictions["prediction_eV"].view(-1).double()
    target = predictions["target_eV"].view(-1).double()
    source_idx = predictions["source_idx"].view(-1).long()
    require(prediction.numel() == target.numel() == source_idx.numel() == DEVELOPMENT_ROWS, "prediction count")
    require(bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()), "finite predictions")
    require(torch.equal(source_idx, torch.arange(100_000, 150_000)), "development order")
    recomputed_mae = float((prediction - target).abs().mean())
    require(
        abs(recomputed_mae - float(completion["best_development_mae_eV"])) <= 1e-7,
        "recomputed MAE",
    )

    require(sha256_file(best_path) == completion["best_model_sha256"], "best model hash")
    require(sha256_file(predictions_path) == completion["development_predictions_sha256"], "prediction hash")
    require(sha256_file(reference_path) == completion["frozen_reference_sha256"], "reference hash")
    require(reference.get("frozen_reference") is True, "reference freeze")
    require(reference.get("physical_batch_per_device") == PHYSICAL_BATCH, "reference batch")
    require(reference.get("tail_batch_policy") == "drop_last", "reference tail policy")
    require(reference.get("result_artifact_sha256") == completion["best_model_sha256"], "reference result hash")
    require(best.get("development_mae_eV") == completion["best_development_mae_eV"], "best metric")
    require(best.get("manifest_sha256") == MANIFEST_SHA256, "model data identity")
    validate_runtime_certificate(preflight["runtime_certificate"], reference)
    for payload in (completion, predictions):
        require(payload.get("official_validation_role_read") is False, "official validation role")
        require(payload.get("test_dev_role_read") is False, "test-dev role")
        require(payload.get("test_challenge_role_read") is False, "test challenge role")

    result = {
        "format": "molgap-pcqm-gptrans-t-100k-reference-acceptance-v4",
        "accepted": True,
        "best_development_mae_eV": recomputed_mae,
        "best_epoch": int(completion["best_epoch"]),
        "runtime_certificate_id": completion["runtime_certificate_id"],
        "manifest_sha256": MANIFEST_SHA256,
        "best_model_sha256": completion["best_model_sha256"],
        "development_predictions_sha256": completion["development_predictions_sha256"],
        "frozen_reference_sha256": completion["frozen_reference_sha256"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(acceptance_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    print(accept(args.output, args.acceptance))


if __name__ == "__main__":
    main()
