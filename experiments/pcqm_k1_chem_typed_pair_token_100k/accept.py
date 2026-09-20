"""No-inference acceptance for the chemistry-conditioned PairToken screen."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_chem_typed_pair_token"
EXPECTED_PARAMETERS = {MODE: 3_682_417}
SIDECAR_SHA256 = "42a40fa186871d2b3c42b67965af85d3372b79c332694eb6b086ca782583f82b"
SIDECAR_CONTRACT_SHA256 = "dde86838f2d8e42d6f8c477bd04278234d7cb02b7240a5395291ce10ae9cf2d8"


def accept(reference_root: Path, candidate_root: Path, source_commit: str, archive_sha256: str):
    shared_path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", shared_path)
    shared = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(shared)
    result = shared.accept(
        reference_root,
        candidate_root,
        modes=(MODE,),
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="nested-function",
    )
    record = result["candidates"][MODE]["record"]
    if record["source_commit"] != source_commit:
        raise RuntimeError("Source commit mismatch")
    checks = record["preflight"]["mechanism_checks"]
    required = {
        "target_layer": 6,
        "pair_channels": 32,
        "role_channels": 16,
        "group_types": 12,
        "pair_source": "all-ordered-pairs-of-current-layer6-node-states",
        "selection_conditioning": "frozen-smarts-atom-role-incidence",
        "relation_tokens": 1,
        "dense_atom_to_atom_attention": False,
        "membership_binary": True,
        "valid_pair_count_exact": True,
        "assignment_mass_one": True,
        "padding_mass_zero": True,
        "zero_role_query": True,
        "chemistry_free_role_zero": True,
        "zero_return_projection": True,
        "zero_initial_update_exact": True,
        "resume_two_step_bitwise_equal": True,
    }
    if any(checks.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Chemistry-conditioned mechanism mismatch: {checks}")
    sidecar = record.get("functional_group_sidecar", {})
    if (
        sidecar.get("aggregate_sha256") != SIDECAR_SHA256
        or sidecar.get("functional_group_contract_sha256") != SIDECAR_CONTRACT_SHA256
        or sidecar.get("gap_labels_read") is not False
        or sidecar.get("protected_roles_read") is not False
    ):
        raise RuntimeError("Functional-group sidecar identity mismatch")
    root = candidate_root / MODE
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("source_archive_sha256") != archive_sha256 or checkpoint.get("epoch") != 39:
        raise RuntimeError("Checkpoint identity or completion mismatch")
    for name, artifact_digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != artifact_digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    result.update(
        {
            "format": "molgap-pcqm-k1-chem-typed-pair-token-acceptance-v1",
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
