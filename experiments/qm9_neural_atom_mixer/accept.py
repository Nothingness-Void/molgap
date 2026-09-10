"""No-inference acceptance for the QM9 Neural-Atom mixer screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.qm9_gape import sha256_file
from molgap.qm9_neural_atom import (
    LATENT_CHANNELS,
    MAX_EPOCH_TIME_RATIO,
    MAX_PARAMETER_COUNT,
    MAX_SLOTS,
    MIN_GAIN_VS_BASELINE_EV,
    MIN_GAIN_VS_ONE_SLOT_EV,
    MIXER_LAYERS,
)
from molgap.screen_policy import validate_paired_screen_contract


def accept(root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "format": "molgap-qm9-neural-atom-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "required_gain_vs_baseline_eV": MIN_GAIN_VS_BASELINE_EV,
        "required_gain_vs_one_slot_eV": MIN_GAIN_VS_ONE_SLOT_EV,
        "max_epoch_time_ratio": MAX_EPOCH_TIME_RATIO,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if metrics.get(key) != value or completion.get(key) != value:
            raise RuntimeError(f"Result contract changed for {key}")
    results = metrics["results"]
    if set(results) != {"full_gps", "neural_atom_k1", "neural_atom_k4"}:
        raise RuntimeError("Unexpected Neural-Atom arm set")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    if comparability != metrics["comparability"]:
        raise RuntimeError("Stored comparability report changed")
    preflight = metrics["preflight"]
    if (
        preflight.get("accepted") is not True
        or preflight.get("mixer_layers") != list(MIXER_LAYERS)
        or preflight.get("max_slots") != MAX_SLOTS
        or preflight.get("latent_channels") != LATENT_CHANNELS
        or preflight.get("test_role_read") is not False
    ):
        raise RuntimeError("Remote architecture preflight changed")
    candidate_arms = preflight["candidate_arms"]
    if (
        candidate_arms["neural_atom_k1"]["parameter_count"]
        != candidate_arms["neural_atom_k4"]["parameter_count"]
        or candidate_arms["neural_atom_k1"]["initial_state_sha256"]
        != candidate_arms["neural_atom_k4"]["initial_state_sha256"]
    ):
        raise RuntimeError("K1/K4 parameter-matched control changed")
    for name, active_slots in (("neural_atom_k1", 1), ("neural_atom_k4", 4)):
        arm = candidate_arms[name]
        if arm["shared_backbone_sha256"] != preflight["baseline_shared_backbone_sha256"]:
            raise RuntimeError(f"Shared backbone changed for {name}")
        if arm["finite_return_projection_gradients"] is not True:
            raise RuntimeError(f"Return gradient failed for {name}")
        for check in arm["mixer_checks"]:
            expected = {
                "active_slots": active_slots,
                "assignment_mass_exact": True,
                "padding_mass_zero": True,
                "zero_return_exact": True,
                "output_projection_zero": True,
            }
            for key, value in expected.items():
                if check.get(key) != value:
                    raise RuntimeError(f"Mixer check changed for {name}/{key}")
    recomputed = {}
    for name, result in results.items():
        training = result["training"]
        if name != "full_gps" and training["parameter_count"] > MAX_PARAMETER_COUNT:
            raise RuntimeError(f"Parameter ceiling failed for {name}")
        payload_path = root / name / "best_validation_payload.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        value = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        if abs(value - training["validation_gap_mae_eV"]) > 1e-9:
            raise RuntimeError(f"Validation metric changed for {name}")
        if sha256_file(payload_path) != training["payload_sha256"]:
            raise RuntimeError(f"Payload hash changed for {name}")
        recomputed[name] = value
    for relative, expected in completion["artifact_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Artifact hash changed: {relative}")
    gain_baseline = recomputed["full_gps"] - recomputed["neural_atom_k4"]
    gain_one_slot = recomputed["neural_atom_k1"] - recomputed["neural_atom_k4"]
    epoch_ratio = (
        results["neural_atom_k4"]["training"]["mean_epoch_seconds"]
        / results["full_gps"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_baseline >= MIN_GAIN_VS_BASELINE_EV
        and gain_one_slot >= MIN_GAIN_VS_ONE_SLOT_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["neural_atom_k4"]["training"]["memory_reserve_fraction"] >= 0.15
    )
    if nominated != metrics["pcqm_transfer_nominated"]:
        raise RuntimeError("Stored Neural-Atom decision changed")
    return {
        "format": "molgap-qm9-neural-atom-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "recomputed_validation_gap_mae_eV": recomputed,
        "candidate_gain_vs_baseline_eV": gain_baseline,
        "candidate_gain_vs_one_slot_eV": gain_one_slot,
        "candidate_epoch_time_ratio": epoch_ratio,
        "pcqm_transfer_nominated": nominated,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }


def main(argv=None):
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
