"""No-inference acceptance for the PCQM-100K Fourier-Edge transfer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.pcqm_fourier_edge import (
    EXPECTED_PARAMETERS,
    GEOMETRY_CACHE_SHA256,
    MAX_EPOCH_TIME_RATIO,
    MIN_GAIN_VS_FULL_GPS_EV,
    MIN_GAIN_VS_K1_EV,
    PARENT_GRAPH_CACHE_SHA256,
)
from molgap.qm9_gape import sha256_file
from molgap.screen_policy import validate_paired_screen_contract


def accept(root: Path, *, source_commit: str) -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "format": "molgap-pcqm-gap100k-fourier-edge-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_CACHE_SHA256,
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "required_gain_vs_full_gps_eV": MIN_GAIN_VS_FULL_GPS_EV,
        "required_gain_vs_k1_eV": MIN_GAIN_VS_K1_EV,
        "max_epoch_time_ratio": MAX_EPOCH_TIME_RATIO,
        "pure_2d": True,
        "model_inference_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_audit_read": False,
    }
    for key, value in required.items():
        if metrics.get(key) != value or completion.get(key) != value:
            raise RuntimeError(f"Result contract changed for {key}")
    results = metrics["results"]
    if set(results) != set(EXPECTED_PARAMETERS):
        raise RuntimeError("Unexpected PCQM Fourier-Edge arm set")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    if comparability != metrics["comparability"]:
        raise RuntimeError("Stored comparability report changed")
    preflight = metrics["preflight"]
    fourier = preflight["arms"]["fourier_edge_k1"]
    if (
        preflight.get("accepted") is not True
        or preflight.get("geometry_attributes_removed_before_batching") is not True
        or preflight.get("official_validation_role_read") is not False
        or preflight.get("test_dev_role_read") is not False
        or fourier.get("manual_equation_match") is not True
        or fourier.get("all_fourier_gradients_nonzero") is not True
    ):
        raise RuntimeError("Remote PCQM Fourier preflight changed")
    if (
        preflight["arms"]["neural_atom_k1"]["shared_without_edge_proposal_sha256"]
        != fourier["shared_without_edge_proposal_sha256"]
    ):
        raise RuntimeError("K1/Fourier shared initialization changed")

    recomputed = {}
    reference_target = None
    for name, result in results.items():
        training = result["training"]
        if training["parameter_count"] != EXPECTED_PARAMETERS[name]:
            raise RuntimeError(f"Parameter count changed for {name}")
        payload_path = root / name / "best_validation_payload.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        target = payload["target_eV"].contiguous()
        prediction = payload["prediction_eV"].contiguous()
        if target.numel() != 10_000 or prediction.shape != target.shape:
            raise RuntimeError(f"Validation payload shape changed for {name}")
        if reference_target is None:
            reference_target = target
        elif not torch.equal(reference_target, target):
            raise RuntimeError("Validation targets differ across paired arms")
        value = float((prediction - target).abs().mean())
        if abs(value - training["validation_gap_mae_eV"]) > 1e-9:
            raise RuntimeError(f"Validation metric changed for {name}")
        if sha256_file(payload_path) != training["payload_sha256"]:
            raise RuntimeError(f"Payload hash changed for {name}")
        recomputed[name] = value
    for relative, expected in completion["artifact_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Artifact hash changed: {relative}")

    gain_full = recomputed["full_gps"] - recomputed["fourier_edge_k1"]
    gain_k1 = recomputed["neural_atom_k1"] - recomputed["fourier_edge_k1"]
    epoch_ratio = (
        results["fourier_edge_k1"]["training"]["mean_epoch_seconds"]
        / results["neural_atom_k1"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_full >= MIN_GAIN_VS_FULL_GPS_EV
        and gain_k1 >= MIN_GAIN_VS_K1_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["fourier_edge_k1"]["training"]["memory_reserve_fraction"] >= 0.15
    )
    if nominated != metrics["shadow_audit_authorized"]:
        raise RuntimeError("Stored transfer decision changed")
    return {
        "format": "molgap-pcqm-gap100k-fourier-edge-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "geometry_cache_aggregate_sha256": GEOMETRY_CACHE_SHA256,
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "comparability": comparability,
        "recomputed_validation_gap_mae_eV": recomputed,
        "candidate_gain_vs_full_gps_eV": gain_full,
        "candidate_gain_vs_k1_eV": gain_k1,
        "candidate_epoch_time_ratio": epoch_ratio,
        "shadow_audit_authorized": nominated,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_audit_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.root, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
