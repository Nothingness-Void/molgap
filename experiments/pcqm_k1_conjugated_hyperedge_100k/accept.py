"""No-inference terminal acceptance for both conjugated hyperedge arms."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_conjugated_hyperedge import LAYERS, MODES
from molgap.training_reproducibility import atomic_json, sha256_file


EXPECTED_PARAMETERS = {mode: 3_679_617 for mode in MODES}


def accept(reference_root, candidate_root, source_commit, archive_sha256, sidecar_sha256):
    shared_path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", shared_path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root, candidate_root, modes=MODES,
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="nested-function",
    )
    for mode in MODES:
        arm = result["candidates"][mode]["record"]
        checks = arm["preflight"]["mechanism_checks"]
        if (
            arm["source_commit"] != source_commit
            or arm["contract"]["source_archive_sha256"] != archive_sha256
            or arm.get("conjugated_sidecar", {}).get("aggregate_sha256") != sidecar_sha256
            or checks.get("component_layers") != list(LAYERS[mode])
            or checks.get("component_channels") != 32
            or checks.get("component_count", 0) <= 0
            or checks.get("zero_return_exact") is not True
            or checks.get("component_ids_valid") is not True
            or checks.get("resume_two_step_bitwise_equal") is not True
            or checks.get("global_slot_unchanged") is not True
            or checks.get("real_bond_edge_state_unchanged") is not True
        ):
            raise RuntimeError(f"Conjugated mechanism or identity invalid: {mode}")
        root = candidate_root / mode
        checkpoint = torch.load(
            root / "last_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if checkpoint.get("source_archive_sha256") != archive_sha256 or checkpoint.get("epoch") != 39:
            raise RuntimeError(f"Checkpoint identity or completion mismatch: {mode}")
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(root / name) != digest:
                raise RuntimeError(f"Selected artifact mismatch: {mode}/{name}")
    result.update({
        "format": "molgap-k1-conjugated-dual-acceptance-v1",
        "source_commit": source_commit,
        "source_archive_sha256": archive_sha256,
        "conjugated_sidecar_aggregate_sha256": sidecar_sha256,
        "model_inference_executed": False,
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(
        args.reference_root, args.candidate_root, args.source_commit,
        args.archive_sha256, args.sidecar_sha256,
    ))
