"""Read-only, structure-defined residual audit of already accepted predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.pcqm_k1_cross_scale_frozen.accept_and_analyze import atomic_json, load_chunks, sha256
from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT
RECORD = ROOT / "platforms/_records/scnet/k1_cross_scale_frozen/job_122911152/output"
MATRIX = ROOT / "platforms/_records/local/pcqm_molecular_router_audit/aligned_prediction_matrix.pt"
MATRIX_SHA = "c52e944def35d8d9ccb41011c9cf820182702c1a9006189b6fc488397169edbb"
EXTRA = {
    "metagin_2d": (
        "platforms/_records/kaggle/training/metagin_2d_s42_v2/molgap-metagin-2d-s42-v2/best_development_payload.pt",
        "experiments/pcqm_metagin_2d_100k/attempt_v2/results/acceptance_summary.json",
    ),
    "motif_k1": (
        "platforms/_records/kaggle/training/motif_hierarchy_gpu_v3/pcqm_k1_motif_hierarchy/neural_atom_k1_motif_hierarchy/best_development_payload.pt",
        "experiments/pcqm_motif_hierarchy_100k/attempt_v3/results/acceptance_summary.json",
    ),
}


def values(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def effect(reference: np.ndarray, candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    gain = np.abs(reference[mask] - target[mask]) - np.abs(candidate[mask] - target[mask])
    return {"rows": int(mask.sum()), "gain_eV": float(gain.mean()),
            "candidate_win_fraction": float((gain > 0).mean())}


def groups(descriptors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    atoms = values(descriptors["atom_count"])
    conjugation = values(descriptors["conjugated_bond_fraction"])
    return {
        "atoms_le_12": atoms <= 12,
        "atoms_13_to_15": (atoms > 12) & (atoms <= 15),
        "atoms_ge_16": atoms > 15,
        "conjugation_le_025": conjugation <= .25,
        "conjugation_025_to_050": (conjugation > .25) & (conjugation <= .5),
        "conjugation_050_to_075": (conjugation > .5) & (conjugation <= .75),
        "conjugation_gt_075": conjugation > .75,
        "atoms_ge_16_and_conjugation_gt_075": (atoms > 15) & (conjugation > .75),
    }


def main() -> None:
    terminal_path = RECORD / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not terminal["complete"] or any(terminal[f"{role}_role_read"] for role in
        ("official_validation", "test_dev", "test_challenge")):
        raise RuntimeError("Cross-scale terminal is incomplete or opened a protected role")
    if sha256(MATRIX) != MATRIX_SHA:
        raise RuntimeError("Accepted 100K alignment matrix changed")
    matrix = torch.load(MATRIX, map_location="cpu", weights_only=True)
    if matrix["format"] != "molgap-pcqm-specialist-prediction-matrix-v1":
        raise RuntimeError("Unexpected aligned matrix format")
    arms = {
        role: {arm: load_chunks(RECORD, terminal, role, arm) for arm in ("k1", "pair_token")}
        for role in ("original_100k", "unseen_500k")
    }
    original = arms["original_100k"]
    if not torch.equal(original["k1"]["source_idx"], matrix["source_idx"]):
        raise RuntimeError("100K row identities differ")
    if not torch.equal(original["k1"]["target_eV"], matrix["target_eV"]):
        raise RuntimeError("100K target identities differ")
    k1_index = matrix["model_ids"].index("neural_atom_k1_v4")
    if not np.allclose(values(original["k1"]["prediction_eV"]),
                       values(matrix["predictions_eV"][:, k1_index]), atol=1e-5, rtol=0):
        raise RuntimeError("K1 predictions failed accepted reproduction tolerance")
    for role, pair in arms.items():
        if not torch.equal(pair["k1"]["source_idx"], pair["pair_token"]["source_idx"]):
            raise RuntimeError(f"{role}: source rows differ")
        if not torch.equal(pair["k1"]["target_eV"], pair["pair_token"]["target_eV"]):
            raise RuntimeError(f"{role}: targets differ")
        for name in pair["k1"]["descriptors"]:
            if not torch.equal(pair["k1"]["descriptors"][name], pair["pair_token"]["descriptors"][name]):
                raise RuntimeError(f"{role}: descriptor {name} differs")
    candidates = {}
    for name, (relative, summary_relative) in EXTRA.items():
        path, summary_path = ROOT / relative, ROOT / summary_relative
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary["accepted"] or sha256(path) != summary["artifact_sha256"]["best_development_payload.pt"]:
            raise RuntimeError(f"{name}: unaccepted or changed prediction payload")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(payload["source_idx"].view(-1).long(), matrix["source_idx"].view(-1).long()):
            raise RuntimeError(f"{name}: source rows differ")
        if not torch.equal(payload["target_eV"].view(-1).float(), matrix["target_eV"].view(-1).float()):
            raise RuntimeError(f"{name}: targets differ")
        candidates[name] = {"prediction": values(payload["prediction_eV"]), "sha256": sha256(path)}
    candidates["pair_token"] = {"prediction": values(original["pair_token"]["prediction_eV"]),
                                "sha256": terminal["identity"]["payload_sha256"]["pair_token"]}
    report = {"format": "molgap-k1-structural-residual-audit-v1", "source_type": "derived_from_accepted_artifacts",
              "new_training": False, "new_model_inference": False, "protected_role_access": False,
              "selection_caveat": "Post-hoc structure groups on repeatedly used internal development roles; not a promotion, router, or new independent validation.",
              "comparison_caveat": "Only K1/PairToken checkpoints are shared across the two roles; MetaGIN2D and motif K1 have original-100K endpoints only.",
              "matrix_sha256": MATRIX_SHA, "cross_scale_terminal_sha256": sha256(terminal_path),
              "models": {}, "role_rows": {}}
    for role, pair in arms.items():
        target, reference = values(pair["k1"]["target_eV"]), values(pair["k1"]["prediction_eV"])
        masks = groups(pair["k1"]["descriptors"])
        report["role_rows"][role] = len(target)
        role_candidates = candidates if role == "original_100k" else {
            "pair_token": {"prediction": values(pair["pair_token"]["prediction_eV"]),
                           "sha256": terminal["identity"]["checkpoint_sha256"]["pair_token"]}}
        for name, entry in role_candidates.items():
            model = report["models"].setdefault(name, {"original_payload_sha256": candidates[name]["sha256"], "roles": {}})
            model["roles"][role] = {
                "overall": effect(reference, entry["prediction"], target, np.ones(len(target), dtype=bool)),
                "structure_groups": {key: effect(reference, entry["prediction"], target, mask)
                                     for key, mask in masks.items() if int(mask.sum()) >= 500},
            }
    output = ROOT / "experiments/pcqm_k1_cross_scale_frozen/results/structural_residual_audit.json"
    atomic_json(output, report)
    print(f"saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
