"""No-inference acceptance for the PairToken 500K bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from molgap.server_acceptance import make_outcome


EXPECTED_MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
EXPECTED_PARAMETERS = 3_681_665
EXPECTED_REFERENCE = 0.10485986978054046


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path) -> dict:
    import torch

    completion = json.loads((root / "completion_manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    contract = json.loads((root / "scientific_contract.json").read_text())
    trace = json.loads((root / "trace.json").read_text())["epochs"]
    preflight = json.loads((root / "preflight.json").read_text())
    payload = torch.load(
        root / "best_predictions.pt", map_location="cpu", weights_only=False
    )

    if completion.get("complete") is not True or metrics.get("complete") is not True:
        raise RuntimeError("Remote run is not complete")
    if contract["data_role_fingerprint"] != EXPECTED_MANIFEST:
        raise RuntimeError("Fixed 500K identity changed")
    for key, expected in {
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "precision": "fp32",
        "epochs": 60,
        "steps_per_epoch": 3906,
        "sample_exposure": 29_998_080,
    }.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Scientific contract changed: {key}")
    if metrics["parameter_count"] != EXPECTED_PARAMETERS:
        raise RuntimeError("PairToken parameter count changed")
    if preflight["memory_reserve_fraction"] < 0.15:
        raise RuntimeError("Preflight memory reserve failed")
    if len(trace) != 60 or [item["epoch"] for item in trace] != list(range(60)):
        raise RuntimeError("Epoch trace is incomplete")
    if trace[-1]["sample_presentations"] != 29_998_080:
        raise RuntimeError("Training exposure changed")
    if not all(math.isfinite(item["development_mae_eV"]) for item in trace):
        raise RuntimeError("Non-finite development metric")
    expected_idx = torch.arange(500_000, 550_000, dtype=torch.long)
    source_idx = payload["source_idx"]
    prediction = payload["prediction"]
    target = payload["target"]
    if source_idx.ndim != 1 or prediction.ndim != 1 or target.ndim != 1:
        raise RuntimeError("Development payload tensors must be one-dimensional")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise RuntimeError("Development predictions or targets are non-finite")
    if not torch.equal(source_idx.long(), expected_idx):
        raise RuntimeError("Development source indices changed")
    if target.numel() != 50_000 or prediction.numel() != 50_000:
        raise RuntimeError("Development payload row count changed")
    recomputed = float((prediction - target).abs().mean())
    if abs(recomputed - metrics["development_gap_mae_eV"]) > 1e-7:
        raise RuntimeError("Saved predictions do not reproduce the reported MAE")
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if metrics.get(key) is not False:
            raise RuntimeError(f"Sealed role flag changed: {key}")
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Artifact hash mismatch: {name}")

    gain = EXPECTED_REFERENCE - recomputed
    return {
        # Kept for callers of the historical mechanical CLI.  V5 consumers must
        # use the separate outcome fields below rather than this compatibility bit.
        "accepted": True,
        "v5_outcome": make_outcome(
            execution_status="COMPLETE",
            artifact_status="ACCEPTED",
            # This no-inference checker has no reference prediction bundle,
            # paired analysis, bootstrap, or native-cost ledger yet.
            comparison_status="PENDING",
            scientific_status="PENDING",
            transfer_status="NOT_READY",
            budget_decision="PENDING",
            full_handoff_status="NONE",
        ),
        "v5_missing_evidence": [
            "reference_prediction_bundle",
            "paired_analysis",
            "paired_bootstrap",
            "actual_native_cost",
        ],
        "model_inference_executed": False,
        "development_gap_mae_eV": recomputed,
        "frozen_k1_reference_mae_eV": EXPECTED_REFERENCE,
        "gain_over_k1_eV": gain,
        "material_gate_passed": gain >= 0.003,
        "best_epoch": metrics["best_epoch"],
        "parameter_count": EXPECTED_PARAMETERS,
        "payload_sha256": sha256_file(root / "best_predictions.pt"),
        "model_sha256": sha256_file(root / "best_model.pt"),
        "checkpoint_sha256": sha256_file(root / "last_checkpoint.pt"),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root.resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
