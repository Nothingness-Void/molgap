"""Saved-artifact acceptance for the GPSPP-local K1 screen."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_gpspp_local import MODES, PARAMETERS
from molgap.training_reproducibility import atomic_json, sha256_file


def accept(reference_root, candidate_root, source_commit, archive_sha256):
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_frozen_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root,
        candidate_root,
        modes=MODES,
        expected_parameters=PARAMETERS,
        initialization_policy="nested-function",
    )
    reference_sha = result["reference"]["preflight"][
        "shared_k1_initial_state_sha256"
    ]
    for mode in MODES:
        root = candidate_root / mode
        candidate = result["candidates"][mode]
        record = candidate["record"]
        preflight = record["preflight"]
        checks = preflight.get("mechanism_checks", {})
        if preflight.get("shared_k1_initial_state_sha256") != reference_sha:
            raise RuntimeError(f"Frozen K1 initialization changed: {mode}")
        if (
            checks.get("equations_verified") is not True
            or checks.get("real_bonds_only") is not True
            or checks.get("directional_endpoint_aggregation") is not True
            or checks.get("adapter_parameters") != 857_088
            or len(checks.get("layers", [])) != 9
            or checks.get("resume_two_step_bitwise_equal") is not True
        ):
            raise RuntimeError(f"Directional mechanism checks changed: {mode}")
        if preflight.get("preflight_memory_reserve_fraction", 0.0) < 0.15:
            raise RuntimeError(f"Preflight memory reserve failed: {mode}")
        expected_receiver = mode == MODES[1]
        if checks.get("receiver_enabled") is not expected_receiver:
            raise RuntimeError(f"Directional arm identity changed: {mode}")
        if not all(
            row.get("use_receiver") is expected_receiver
            and row.get("zero_return_exact") is True
            and row.get("output_projection_zero") is True
            and row.get("message_finite") is True
            and row.get("message_nonzero") is True
            and row.get("sender_nonzero") is True
            for row in checks["layers"]
        ):
            raise RuntimeError(f"Directional layer invariant failed: {mode}")
        if record.get("source_commit") != source_commit:
            raise RuntimeError(f"Source commit mismatch: {mode}")
        checkpoint = torch.load(
            root / "last_checkpoint.pt", map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("source_commit") != source_commit
            or checkpoint.get("source_archive_sha256") != archive_sha256
            or checkpoint.get("epoch") != 39
        ):
            raise RuntimeError(f"Checkpoint identity changed: {mode}")
        for key in (
            "optimizer",
            "scheduler",
            "rng_state",
            "best",
            "best_epoch",
            "best_artifact_sha256",
        ):
            if key not in checkpoint:
                raise RuntimeError(f"Incomplete checkpoint: {mode}/{key}")
        if not all(
            key in checkpoint["rng_state"]
            for key in ("python", "numpy", "torch", "cuda")
        ):
            raise RuntimeError(f"Incomplete RNG state: {mode}")
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(root / name) != digest:
                raise RuntimeError(f"Selected artifact mismatch: {mode}/{name}")
        for name in ("best_model.pt", "last_checkpoint.pt"):
            saved = torch.load(root / name, map_location="cpu", weights_only=False)
            state = saved["model"] if name == "last_checkpoint.pt" else saved
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise RuntimeError(f"Nonfinite saved tensor: {mode}/{name}")
        trace = json.loads((root / "trace.json").read_text())["epochs"]
        expected_steps = [781 * (epoch + 1) for epoch in range(40)]
        if [row["optimizer_steps"] for row in trace] != expected_steps:
            raise RuntimeError(f"Optimizer trace mismatch: {mode}")
        if not all(
            (root / f"recovery_epoch_{epoch:02d}.tar").is_file()
            for epoch in (10, 20, 30, 40)
        ):
            raise RuntimeError(f"Recovery chunk missing: {mode}")
    sender = result["candidates"][MODES[0]][
        "recomputed_development_gap_mae_eV"
    ]
    bidirectional = result["candidates"][MODES[1]][
        "recomputed_development_gap_mae_eV"
    ]
    result.update(
        {
            "format": "molgap-k1-gpspp-local-acceptance-v1",
            "scientific_round": 3,
            "source_commit": source_commit,
            "source_archive_sha256": archive_sha256,
            "bidirectional_beats_sender_control": bidirectional < sender,
            "model_inference_executed": False,
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(
        args.output,
        accept(
            args.reference_root,
            args.candidate_root,
            args.source_commit,
            args.archive_sha256,
        ),
    )
