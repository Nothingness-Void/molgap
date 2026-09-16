"""No-inference acceptance for the frozen K1 PairToken candidate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_pair_token"
EXPECTED_PARAMETERS = {MODE: 3_681_665}


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
        initialization_policy="nested-function",
    )


def accept(
    reference_root: Path,
    candidate_root: Path,
    source_commit: str,
    archive_sha256: str,
) -> dict:
    result = _shared_accept(reference_root, candidate_root)
    candidate = result["candidates"][MODE]["record"]
    if candidate["source_commit"] != source_commit:
        raise RuntimeError("Source commit mismatch")
    checks = candidate["preflight"]["mechanism_checks"]
    expected = {
        "target_layer": 6,
        "pair_channels": 32,
        "relation_tokens": 1,
        "pair_source": "all-ordered-pairs-of-current-layer6-node-states",
        "pair_normalization": "per-pair-across-channels",
        "dense_atom_to_atom_attention": False,
        "valid_pair_count_exact": True,
        "assignment_mass_one": True,
        "padding_mass_zero": True,
        "zero_return_projection": True,
        "zero_initial_update_exact": True,
    }
    if checks != expected:
        raise RuntimeError(f"PairToken mechanism mismatch: {checks}")
    root = candidate_root / MODE
    checkpoint = torch.load(
        root / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("source_archive_sha256") != archive_sha256
        or checkpoint.get("source_commit") != source_commit
        or checkpoint.get("epoch") != 39
    ):
        raise RuntimeError("Checkpoint identity or completion mismatch")
    for name in (
        "optimizer",
        "scheduler",
        "rng_state",
        "best",
        "best_epoch",
        "best_artifact_sha256",
    ):
        if name not in checkpoint:
            raise RuntimeError(f"Recoverable checkpoint field missing: {name}")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    result["format"] = "molgap-pcqm-k1-pair-token-acceptance-v1"
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
        accept(
            args.reference_root,
            args.candidate_root,
            args.source_commit,
            args.archive_sha256,
        ),
    )
