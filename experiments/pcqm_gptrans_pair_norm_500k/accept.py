"""No-inference mechanical acceptance for the 500K PairNorm bridge."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from molgap.pcqm_500k_v4_evidence import PARAMETERS, scientific_contract
from molgap.screen_policy import canonical_fingerprint, validate_runtime_certificate
from molgap.training_reproducibility import sha256_file


ARM = "gptrans_pair_update_norm"


def accept(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "stage_manifest.json").read_text())
    if manifest["arm"] != ARM or manifest["parameters"] != PARAMETERS[ARM]:
        raise ValueError("Candidate model identity changed")
    status = manifest["status"]
    next_epoch = manifest["next_epoch"]
    if status == "COMPLETE" and next_epoch != 60:
        raise ValueError("Complete run did not reach epoch 60")
    if status == "FUTILITY_STOPPED" and next_epoch not in {30, 40}:
        raise ValueError("Futility stop occurred outside a registered gate")
    if status not in {"COMPLETE", "FUTILITY_STOPPED", "STAGE_COMPLETE"}:
        raise ValueError(f"Unacceptable terminal status: {status}")
    for role in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if manifest[role] is not False:
            raise ValueError(f"Sealed role accessed: {role}")
    required = {
        "last_checkpoint.pt",
        "best_model.pt",
        "best_predictions.pt",
        "initial_state.pt",
        "trace.json",
        "runtime_certificate.json",
        "calibration.json",
    }
    if status == "FUTILITY_STOPPED":
        required.add("futility_decisions.json")
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("Required durable artifacts are missing")
    for name, checksum in manifest["artifacts"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        if sha256_file(path) != checksum:
            raise ValueError(f"Artifact hash mismatch: {name}")
    # Training binds the frozen transform rule to the measured train-only mean/std.
    best_model = torch.load(root / "best_model.pt", map_location="cpu", weights_only=True)
    mean, std = best_model["mean"], best_model["std"]
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise ValueError("Invalid train-only target transform")
    realized_contract = scientific_contract(ARM)
    realized_contract["target_transform_fingerprint"] = canonical_fingerprint(
        {"mean": mean, "std": std}
    )
    if manifest["contract"] != realized_contract or best_model["contract"] != realized_contract:
        raise ValueError("Candidate scientific contract changed")
    trace = json.loads((root / "trace.json").read_text())["epochs"]
    if [row["epoch"] for row in trace] != list(range(next_epoch)):
        raise ValueError("Training trace is incomplete or reordered")
    certificate = json.loads((root / "runtime_certificate.json").read_text())
    validate_runtime_certificate(
        certificate,
        {
            "platform_id": certificate["platform_id"],
            "accelerator": certificate["accelerator"],
            "runtime_certificate_id": manifest["runtime_certificate_id"],
        },
    )
    return {
        "accepted": True,
        "status": status,
        "next_epoch": next_epoch,
        "best_development_mae_eV": manifest["best_development_mae_eV"],
        "best_epoch": manifest["best_epoch"],
        "contract_fingerprint": canonical_fingerprint(manifest["contract"]),
        "manifest_sha256": sha256_file(root / "stage_manifest.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(accept(arguments.root), indent=2))

