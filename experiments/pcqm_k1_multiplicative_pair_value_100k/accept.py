"""No-inference terminal artifact acceptance for the frozen candidate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_multiplicative_pair_value"
EXPECTED_PARAMETERS = {MODE: 3_694_081}


def accept(reference_root: Path, candidate_root: Path, source_commit: str, archive_sha256: str):
    shared_path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", shared_path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root,
        candidate_root,
        modes=(MODE,),
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="nested-function",
    )
    candidate = result["candidates"][MODE]["record"]
    if candidate["source_commit"] != source_commit:
        raise RuntimeError("Source commit mismatch")
    checks = candidate["preflight"]["mechanism_checks"]
    required = {
        "selector_value_parameters_disjoint": True,
        "valid_pair_count_exact": True,
        "assignment_mass_one": True,
        "padding_mass_zero": True,
        "zero_return_projection": True,
        "zero_initial_update_exact": True,
        "pair_selection": "accepted-additive-learned-query",
        "pair_value": "separate-ordered-low-rank-hadamard-product",
    }
    if any(checks.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Multiplicative PairValue mechanism mismatch: {checks}")
    root = candidate_root / MODE
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("source_archive_sha256") != archive_sha256 or checkpoint.get("epoch") != 39:
        raise RuntimeError("Checkpoint identity or completion mismatch")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    result.update(
        {
            "format": "molgap-pcqm-k1-multiplicative-pair-value-acceptance-v1",
            "source_commit": source_commit,
            "source_archive_sha256": archive_sha256,
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
        accept(args.reference_root, args.candidate_root, args.source_commit, args.archive_sha256),
    )
