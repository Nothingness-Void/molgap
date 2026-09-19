"""No-inference acceptance for the frozen recurrent-pair candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_recurrent_pair_bridge"
EXPECTED_PARAMETERS = 3_689_698
EXPECTED_SOURCE_COMMIT = "b1e4790f240c578671ad78210749a4970ba4780d"
EXPECTED_REFERENCE_PAYLOAD_SHA256 = (
    "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91"
)
MINIMUM_GAIN_EV = 0.003


def _bootstrap_interval(delta: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(42)
    draws = []
    for _ in range(2_000):
        indices = generator.integers(0, len(delta), size=len(delta))
        draws.append(float(delta[indices].mean()))
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def accept(candidate_root: Path, reference_payload: Path) -> dict:
    root = candidate_root / MODE
    completion = json.loads((root / "completion_manifest.json").read_text())
    record = json.loads((root / "arm_record.json").read_text())
    if completion.get("complete") is not True or completion.get("mode") != MODE:
        raise RuntimeError("Candidate completion identity changed")
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Candidate artifact hash mismatch: {name}")
    training = record["training"]
    if (
        record.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or training.get("parameter_count") != EXPECTED_PARAMETERS
        or training.get("epochs_completed") != 40
        or training.get("optimizer_steps") != 31_240
        or training.get("sample_presentations") != 3_998_720
    ):
        raise RuntimeError("Candidate identity, parameter count, or exposure changed")
    if any(
        record.get(field) is not False
        for field in (
            "official_validation_role_read",
            "test_dev_role_read",
            "test_challenge_role_read",
        )
    ):
        raise RuntimeError("A protected role was read")
    checks = record["preflight"]["mechanism_checks"]
    if (
        checks.get("exchange_layers") != [3, 6, 9]
        or checks.get("pair_channels") != 32
        or checks.get("return_projections_zero") is not True
        or checks.get("resume_two_step_bitwise_equal") is not True
        or not all(
            layer.get("recurrent_addition_exact") is True
            and layer.get("valid_target_mass_one") is True
            and layer.get("padding_assignment_zero") is True
            and layer.get("padding_pair_state_zero") is True
            and layer.get("zero_initial_update_exact") is True
            for layer in checks.get("layers", [])
        )
    ):
        raise RuntimeError(f"Recurrent-pair mechanism mismatch: {checks}")

    if sha256_file(reference_payload) != EXPECTED_REFERENCE_PAYLOAD_SHA256:
        raise RuntimeError("Frozen K1 reference payload changed")
    reference = torch.load(reference_payload, map_location="cpu", weights_only=False)
    candidate = torch.load(
        root / "best_development_payload.pt", map_location="cpu", weights_only=False
    )
    for field in ("source_idx", "target_eV"):
        if not torch.equal(reference[field], candidate[field]):
            raise RuntimeError(f"Reference/candidate {field} alignment changed")
    reference_error = (reference["prediction_eV"] - reference["target_eV"]).abs()
    candidate_error = (candidate["prediction_eV"] - candidate["target_eV"]).abs()
    reference_mae = float(reference_error.mean())
    candidate_mae = float(candidate_error.mean())
    gain = reference_mae - candidate_mae
    interval = _bootstrap_interval(
        (candidate_error - reference_error).numpy().astype(np.float64)
    )
    selected = gain >= MINIMUM_GAIN_EV and interval[1] < 0.0
    return {
        "format": "molgap-pcqm-k1-recurrent-pair-acceptance-v1",
        "accepted": True,
        "model_inference_executed": False,
        "candidate_id": MODE,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "reference_payload_sha256": EXPECTED_REFERENCE_PAYLOAD_SHA256,
        "candidate_payload_sha256": sha256_file(
            root / "best_development_payload.pt"
        ),
        "reference_mae_eV": reference_mae,
        "candidate_mae_eV": candidate_mae,
        "gain_eV": gain,
        "candidate_minus_reference_bootstrap_95_eV": list(interval),
        "minimum_gain_eV": MINIMUM_GAIN_EV,
        "selected_candidate": MODE if selected else None,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(
        args.output,
        accept(args.candidate_root, args.reference_payload),
    )
