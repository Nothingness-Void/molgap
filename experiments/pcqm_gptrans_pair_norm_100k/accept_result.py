"""No-inference acceptance for the GPTrans pair-normalization screen."""
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
from molgap.router import paired_bootstrap_mean
from molgap.screen_policy import evaluate_reference_gain, validate_runtime_certificate
from molgap.training_reproducibility import atomic_json, sha256_file


MODES = ("pair_update_norm", "pair_post_norm")
EXPECTED_PARAMETERS = 5_246_817
EXPECTED_SOURCE_COMMIT = "5f8d27e52d73dca782fad19cde63896e590e8245"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "8b034350a7188cccc3c9b112625d874af910bc8034ea6591f1607f8e85d47ccf"
)
EXPECTED_REFERENCE_PREDICTIONS_SHA256 = (
    "4fa3d32f83b183503bfafeee33b7b44dc7ee5f396bef5b1b546c956669ce6d29"
)
EXPECTED_REFERENCE_MAE_EV = 0.15662720430791377
MINIMUM_GAIN_EV = 0.003


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_predictions(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prediction = payload["prediction_eV"].view(-1).double()
    target = payload["target_eV"].view(-1).double()
    source_idx = payload["source_idx"].view(-1).long()
    require(
        prediction.numel() == target.numel() == source_idx.numel() == DEVELOPMENT_ROWS,
        f"prediction count: {path}",
    )
    require(bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()), f"finite: {path}")
    require(torch.equal(source_idx, torch.arange(100_000, 150_000)), f"row order: {path}")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        require(payload.get(role) is False, f"sealed role {role}: {path}")
    return {"payload": payload, "prediction": prediction, "target": target, "source_idx": source_idx}


def accept_arm(root: Path, mode: str, reference: dict) -> dict:
    arm = root / mode
    paths = {
        name: arm / name
        for name in (
            "completion_manifest.json",
            "frozen_reference.json",
            "best_model.pt",
            "development_predictions.pt",
            "preflight.json",
            "trace.json",
        )
    }
    for path in paths.values():
        require(path.is_file(), f"missing artifact: {path}")

    completion = json.loads(paths["completion_manifest.json"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["frozen_reference.json"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["preflight.json"].read_text(encoding="utf-8"))
    trace = json.loads(paths["trace.json"].read_text(encoding="utf-8"))["rows"]
    best = torch.load(paths["best_model.pt"], map_location="cpu", weights_only=False)
    predictions = load_predictions(paths["development_predictions.pt"])

    require(completion.get("format") == RUN_FORMAT and completion.get("complete") is True, f"completion: {mode}")
    require(completion.get("variant") == mode and preflight.get("variant") == mode, f"variant: {mode}")
    require(completion.get("parameters") == EXPECTED_PARAMETERS, f"parameters: {mode}")
    require(preflight.get("parameters") == EXPECTED_PARAMETERS and preflight.get("accepted") is True, f"preflight: {mode}")
    require(completion.get("epochs") == EPOCHS and len(trace) == EPOCHS, f"epochs: {mode}")
    require(completion.get("optimizer_steps") == BATCHES_PER_EPOCH * EPOCHS, f"steps: {mode}")
    require(completion.get("sample_presentations") == SAMPLE_PRESENTATIONS, f"exposure: {mode}")
    require(completion.get("manifest_sha256") == MANIFEST_SHA256, f"manifest: {mode}")
    require(completion.get("source_commit") == EXPECTED_SOURCE_COMMIT, f"source commit: {mode}")
    require(completion.get("source_archive_sha256") == EXPECTED_SOURCE_ARCHIVE_SHA256, f"source archive: {mode}")
    require([int(row["epoch"]) for row in trace] == list(range(EPOCHS)), f"trace order: {mode}")
    require(
        all(int(row["sample_presentations"]) == BATCHES_PER_EPOCH * PHYSICAL_BATCH for row in trace),
        f"epoch exposure: {mode}",
    )

    mae = float((predictions["prediction"] - predictions["target"]).abs().mean())
    require(abs(mae - float(completion["best_development_mae_eV"])) <= 1e-7, f"MAE: {mode}")
    require(sha256_file(paths["best_model.pt"]) == completion["best_model_sha256"], f"model hash: {mode}")
    require(
        sha256_file(paths["development_predictions.pt"]) == completion["development_predictions_sha256"],
        f"prediction hash: {mode}",
    )
    require(sha256_file(paths["frozen_reference.json"]) == completion["frozen_reference_sha256"], f"reference hash: {mode}")
    for name, digest in completion["checkpoint_chunks"].items():
        require(sha256_file(arm / name) == digest, f"checkpoint hash {name}: {mode}")
    require(frozen.get("frozen_reference") is True, f"frozen identity: {mode}")
    require(frozen.get("physical_batch_per_device") == PHYSICAL_BATCH, f"batch: {mode}")
    require(frozen.get("device_count") == 1, f"device count: {mode}")
    require(frozen.get("gradient_accumulation_steps") == 1, f"accumulation: {mode}")
    require(frozen.get("result_artifact_sha256") == completion["best_model_sha256"], f"result hash: {mode}")
    require(best.get("development_mae_eV") == completion["best_development_mae_eV"], f"best metric: {mode}")
    require(best.get("manifest_sha256") == MANIFEST_SHA256, f"best data identity: {mode}")
    validate_runtime_certificate(preflight["runtime_certificate"], frozen)
    for payload in (completion, predictions["payload"]):
        for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
            require(payload.get(role) is False, f"sealed role {role}: {mode}")

    require(torch.equal(predictions["source_idx"], reference["source_idx"]), f"reference rows: {mode}")
    require(torch.equal(predictions["target"], reference["target"]), f"reference targets: {mode}")
    delta = (
        (predictions["prediction"] - predictions["target"]).abs()
        - (reference["prediction"] - reference["target"]).abs()
    ).numpy()
    bootstrap = paired_bootstrap_mean(delta, n_bootstrap=10_000, seed=42)
    gate = evaluate_reference_gain(
        reference_mae_eV=EXPECTED_REFERENCE_MAE_EV,
        candidate_mae_eV=mae,
        stochasticity_floor_eV=MINIMUM_GAIN_EV,
        minimum_material_gain_eV=MINIMUM_GAIN_EV,
        paired_row_bootstrap_upper_eV=float(bootstrap["ci95"][1]),
    )
    return {
        "accepted": True,
        "variant": mode,
        "parameters": EXPECTED_PARAMETERS,
        "best_development_mae_eV": mae,
        "best_epoch": int(completion["best_epoch"]),
        "gain_vs_reference_eV": EXPECTED_REFERENCE_MAE_EV - mae,
        "paired_candidate_minus_reference": bootstrap,
        "shortlist_gate": gate,
        "runtime_certificate_id": completion["runtime_certificate_id"],
        "best_model_sha256": completion["best_model_sha256"],
        "development_predictions_sha256": completion["development_predictions_sha256"],
    }


def accept(root: Path, reference_path: Path, output: Path) -> dict:
    require(sha256_file(reference_path) == EXPECTED_REFERENCE_PREDICTIONS_SHA256, "reference prediction identity")
    reference = load_predictions(reference_path)
    reference_mae = float((reference["prediction"] - reference["target"]).abs().mean())
    require(abs(reference_mae - EXPECTED_REFERENCE_MAE_EV) <= 1e-7, "reference MAE")
    arms = {mode: accept_arm(root, mode, reference) for mode in MODES}
    result = {
        "format": "molgap-gptrans-pair-normalization-100k-v5-acceptance-v1",
        "accepted": True,
        "reference_development_mae_eV": reference_mae,
        "reference_predictions_sha256": EXPECTED_REFERENCE_PREDICTIONS_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "arms": arms,
        "shortlisted": [mode for mode, evidence in arms.items() if evidence["shortlist_gate"]["passed"]],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(accept(args.root, args.reference_predictions, args.output), indent=2))


if __name__ == "__main__":
    main()
