"""No-inference acceptance for the prospective matched60-v4 K1 reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_MANIFEST = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
EXPECTED_PARAMETERS = 3_658_817


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
        raise RuntimeError("Reference run is not complete")
    expected = {
        "data_role_fingerprint": EXPECTED_MANIFEST,
        "model_mode": "neural_atom_k1",
        "parameter_count": EXPECTED_PARAMETERS,
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "precision": "fp32",
        "epochs": 60,
        "steps_per_epoch": 3906,
        "sample_exposure": 29_998_080,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"Reference contract changed: {key}")
    if metrics.get("mode") != "neural_atom_k1":
        raise RuntimeError("Reference model identity changed")
    if metrics.get("parameter_count") != EXPECTED_PARAMETERS:
        raise RuntimeError("Reference parameter count changed")
    if metrics.get("frozen_k1_reference_mae_eV") is not None:
        raise RuntimeError("Reference run must not compare against itself")
    if metrics.get("gain_over_k1_eV") is not None:
        raise RuntimeError("Reference run emitted a self-gain")
    if preflight["memory_reserve_fraction"] < 0.15:
        raise RuntimeError("Preflight memory reserve failed")
    if preflight.get("mechanism") is not None:
        raise RuntimeError("Reference preflight contains candidate mechanism output")
    if len(trace) != 60 or [item["epoch"] for item in trace] != list(range(60)):
        raise RuntimeError("Reference epoch trace is incomplete")
    if trace[-1]["sample_presentations"] != 29_998_080:
        raise RuntimeError("Reference training exposure changed")
    if not all(math.isfinite(item["development_mae_eV"]) for item in trace):
        raise RuntimeError("Reference development trace is non-finite")

    source_idx = payload["source_idx"].long()
    prediction = payload["prediction"]
    target = payload["target"]
    if not torch.equal(source_idx, torch.arange(500_000, 550_000)):
        raise RuntimeError("Reference development row order changed")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise RuntimeError("Reference prediction payload is non-finite")
    recomputed = float((prediction - target).abs().mean())
    if abs(recomputed - metrics["development_gap_mae_eV"]) > 1e-7:
        raise RuntimeError("Reference payload does not reproduce its MAE")
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if metrics.get(key) is not False:
            raise RuntimeError(f"Protected-role flag changed: {key}")
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Reference artifact hash mismatch: {name}")

    return {
        "accepted": True,
        "model_inference_executed": False,
        "mode": "neural_atom_k1",
        "development_gap_mae_eV": recomputed,
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
