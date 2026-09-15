"""No-inference acceptance for the K1 edge-conditioned slot candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_edge_conditioned_slot"
EXPECTED_PARAMETERS = {MODE: 3_671_105}


def _shared_accept(reference_root: Path, candidate_root: Path) -> dict:
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    return shared.accept(
        reference_root,
        candidate_root,
        modes=(MODE,),
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="identical-tensors-altered-edge-dataflow",
    )


def accept(reference_root: Path, candidate_root: Path, source_commit: str, archive_sha256: str) -> dict:
    result = _shared_accept(reference_root, candidate_root)
    candidate = result["candidates"][MODE]["record"]
    if candidate["source_commit"] != source_commit:
        raise RuntimeError("Source commit mismatch")
    if candidate["architecture"].get("geometry") is not False:
        raise RuntimeError("Candidate unexpectedly exposes geometry")
    checks = candidate["preflight"]["mechanism_checks"]
    if checks.get("edge_context_source") != "mean-incident-directed-real-bond-state":
        raise RuntimeError("Edge context source mismatch")
    if checks.get("zero_initialized_edge_key") is not True or len(checks.get("slot_layers", [])) != 3:
        raise RuntimeError("Edge-conditioned slot preflight incomplete")
    root = candidate_root / MODE
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("source_archive_sha256") != archive_sha256 or checkpoint.get("source_commit") != source_commit:
        raise RuntimeError("Checkpoint source identity mismatch")
    if checkpoint.get("epoch") != 39:
        raise RuntimeError("Candidate did not complete 40 epochs")
    for name in ("optimizer", "scheduler", "rng_state", "best", "best_epoch", "best_artifact_sha256"):
        if name not in checkpoint:
            raise RuntimeError(f"Recoverable checkpoint field missing: {name}")
    if not all(key in checkpoint["rng_state"] for key in ("python", "numpy", "torch", "cuda")):
        raise RuntimeError("RNG recovery state incomplete")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    result["format"] = "molgap-pcqm-k1-edge-conditioned-slot-acceptance-v1"
    result["source_commit"] = source_commit
    result["source_archive_sha256"] = archive_sha256
    result["model_inference_executed"] = False
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
        accept(args.reference_root, args.candidate_root, args.source_commit, args.archive_sha256),
    )
