"""Round-2 aligned residual analysis for matched60 K1 versus PairToken."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.router import paired_bootstrap_mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"prediction", "target", "source_idx"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"Invalid prediction payload: {path}")
    return (
        payload["prediction"].view(-1).float(),
        payload["target"].view(-1).float(),
        payload["source_idx"].view(-1).long(),
    )


def summarize_slice(
    name: str,
    mask: np.ndarray,
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
) -> dict:
    gain = reference_error[mask] - candidate_error[mask]
    return {
        "slice": name,
        "rows": int(mask.sum()),
        "reference_mae_eV": float(reference_error[mask].mean()),
        "candidate_mae_eV": float(candidate_error[mask].mean()),
        "candidate_gain_eV": float(gain.mean()),
        "candidate_win_rate": float((gain > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-100k", type=Path, required=True)
    parser.add_argument("--candidate-100k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ref_pred, ref_target, ref_source = load_predictions(args.reference)
    cand_pred, cand_target, cand_source = load_predictions(args.candidate)
    if not torch.equal(ref_source, cand_source) or not torch.equal(ref_target, cand_target):
        raise RuntimeError("500K reference/candidate rows or targets differ")
    if not torch.equal(ref_source, torch.arange(500_000, 550_000)):
        raise RuntimeError("Unexpected 500K development identity")

    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    development = [item for item in manifest["geometry_shards"] if item["role"] == "development"]
    if len(development) != 1:
        raise RuntimeError("Expected one 500K development shard")
    shard = args.cache_root / development[0]["file"]
    if sha256(shard) != development[0]["sha256"]:
        raise RuntimeError("500K development shard hash changed")
    _, slices = torch.load(shard, map_location="cpu", weights_only=False)
    atom_count = (slices["x"][1:] - slices["x"][:-1]).numpy()

    target = ref_target.numpy().astype(np.float64)
    reference_error = np.abs(ref_pred.numpy().astype(np.float64) - target)
    candidate_error = np.abs(cand_pred.numpy().astype(np.float64) - target)
    gain = reference_error - candidate_error
    bootstrap = paired_bootstrap_mean(-gain, n_bootstrap=10_000, seed=42)

    slices_out = []
    for descriptor, values in {
        "atom_count": atom_count,
        "reference_abs_error": reference_error,
    }.items():
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
        groups = np.digitize(values, edges[1:-1], right=True)
        for group in range(len(edges) - 1):
            slices_out.append(
                summarize_slice(
                    f"{descriptor}_q{group + 1}",
                    groups == group,
                    reference_error,
                    candidate_error,
                )
            )

    ref100 = torch.load(args.reference_100k, map_location="cpu", weights_only=True)
    cand100 = torch.load(args.candidate_100k, map_location="cpu", weights_only=True)
    if not torch.equal(ref100["source_idx"], cand100["source_idx"]) or not torch.equal(
        ref100["target_eV"], cand100["target_eV"]
    ):
        raise RuntimeError("100K reference/candidate rows or targets differ")
    y100 = ref100["target_eV"].float()
    gain100 = float(
        (ref100["prediction_eV"].float() - y100).abs().mean()
        - (cand100["prediction_eV"].float() - y100).abs().mean()
    )

    result = {
        "format": "molgap-k1-pair-token-scale-attribution-round2-v1",
        "complete": True,
        "training_executed": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "rows": int(len(target)),
        "row_identity_exact": True,
        "target_identity_exact": True,
        "reference_mae_eV": float(reference_error.mean()),
        "candidate_mae_eV": float(candidate_error.mean()),
        "candidate_gain_eV": float(gain.mean()),
        "candidate_minus_reference_eV": float(-gain.mean()),
        "candidate_win_rate": float((gain > 0).mean()),
        "mean_winning_margin_eV": float(gain[gain > 0].mean()),
        "mean_losing_margin_eV": float((-gain[gain < 0]).mean()),
        "paired_bootstrap_candidate_minus_reference": bootstrap,
        "gain_100k_eV": gain100,
        "gain_retention_500k_over_100k": float(gain.mean() / gain100),
        "material_gain_collapse": bool(gain100 >= 0.003 and gain.mean() < 0.0015),
        "round3_frozen_intervention_gate": bool(gain100 >= 0.003 and gain.mean() < 0.0015),
        "strata": slices_out,
        "artifacts": {
            "reference": {"path": args.reference.as_posix(), "sha256": sha256(args.reference)},
            "candidate": {"path": args.candidate.as_posix(), "sha256": sha256(args.candidate)},
            "cache_manifest": {"path": (args.cache_root / "manifest.json").as_posix(), "sha256": sha256(args.cache_root / "manifest.json")},
            "reference_100k": {"path": args.reference_100k.as_posix(), "sha256": sha256(args.reference_100k)},
            "candidate_100k": {"path": args.candidate_100k.as_posix(), "sha256": sha256(args.candidate_100k)},
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"candidate_gain_eV": result["candidate_gain_eV"], "round3": result["round3_frozen_intervention_gate"]}))


if __name__ == "__main__":
    main()
