"""No-inference terminal artifact acceptance for the frozen candidate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_pair_token_oneshot_triplet"
EXPECTED_PARAMETERS = {MODE: 3_694_753}


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
        "valid_pair_count_exact": True,
        "assignment_mass_one": True,
        "padding_mass_zero": True,
        "adjacency_exact": True,
        "non_backtracking_exact": True,
        "center_identity_exact": True,
        "wedge_count_exact": True,
        "wedge_set_exact": True,
        "message_targets_outgoing_edge": True,
        "zero_triplet_return": True,
        "zero_initial_triplet_update_exact": True,
        "zero_pairtoken_return": True,
        "interaction": "one-shot-incoming-edge-to-outgoing-edge",
        "persistent_triplet_state": False,
        "direct_node_return": False,
        "topology_only": True,
    }
    if any(checks.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"One-shot triplet PairToken mechanism mismatch: {checks}")
    root = candidate_root / MODE
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("source_archive_sha256") != archive_sha256 or checkpoint.get("epoch") != 39:
        raise RuntimeError("Checkpoint identity or completion mismatch")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    result.update(
        {
            "format": "molgap-pcqm-k1-oneshot-triplet-pair-token-acceptance-v1",
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
