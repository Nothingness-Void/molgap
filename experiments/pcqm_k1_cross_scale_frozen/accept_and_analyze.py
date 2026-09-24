"""No-inference acceptance and paired analysis of the frozen cross-scale run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_k1_cross_scale_diagnostic import (
    ARMS, CHUNK_ROWS, DEVELOPMENT, MANIFESTS, TRANSFORM_SHA256,
)
from molgap.router import paired_bootstrap_mean


MATCHED_PAYLOAD_SHA256 = {
    "k1": "e43728a30a74a19e9e969cef3e94f3c0f9718a1c606bfa5306bd9b3c87ca8d31",
    "pair_token": "35fdaa76f10e166ccb123b6005929a420f18b2678a286023d4aa62e48ec9ca01",
}
SOURCE_COMMIT = "ff38382664dcf2261facf6e339106810936c34af"
SOURCE_ARCHIVE_SHA256 = "f08e8991776b43a36b9a9b5128ad7376ff9d15dc4f57e2969594d870080fd9da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_chunks(output_root: Path, terminal: dict, role: str, arm: str) -> dict:
    start, stop, _ = DEVELOPMENT[role]
    rows = []
    for offset in range(0, stop - start, CHUNK_ROWS):
        name = f"{role}/{arm}/chunks/chunk_{offset // CHUNK_ROWS:02d}.pt"
        path = output_root / name
        if name not in terminal["chunk_sha256"] or sha256(path) != terminal["chunk_sha256"][name]:
            raise RuntimeError(f"Missing or corrupt chunk {name}")
        row = torch.load(path, map_location="cpu", weights_only=False)
        expected = torch.arange(start + offset, start + offset + CHUNK_ROWS)
        if not torch.equal(row["source_idx"].view(-1).long(), expected):
            raise RuntimeError(f"Source indices differ in {name}")
        if any(not bool(torch.isfinite(row[key]).all()) for key in ("target_eV", "prediction_eV")):
            raise RuntimeError(f"Nonfinite target or prediction in {name}")
        rows.append(row)
    joined = {
        key: torch.cat([row[key].view(-1) for row in rows])
        for key in ("source_idx", "target_eV", "prediction_eV")
    }
    joined["descriptors"] = {
        key: torch.cat([row["descriptors"][key].view(-1) for row in rows])
        for key in rows[0]["descriptors"]
    }
    if len(joined["source_idx"]) != stop - start:
        raise RuntimeError("Chunk coverage incomplete")
    actual = float((joined["prediction_eV"] - joined["target_eV"]).abs().mean())
    if abs(actual - terminal["records"][f"{role}/{arm}"]["mae_eV"]) > 1e-7:
        raise RuntimeError(f"Terminal MAE differs for {role}/{arm}")
    return joined


def gain_summary(reference: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> dict:
    ref_error = np.abs(reference - target)
    cand_error = np.abs(candidate - target)
    gain = ref_error - cand_error
    return {
        "reference_mae_eV": float(ref_error.mean()),
        "candidate_mae_eV": float(cand_error.mean()),
        "candidate_gain_eV": float(gain.mean()),
        "candidate_row_win_rate": float((gain > 0).mean()),
        "paired_candidate_minus_reference": paired_bootstrap_mean(-gain, n_bootstrap=10_000, seed=42),
    }


def quintile_strata(descriptors: dict[str, torch.Tensor], frozen_gain: np.ndarray,
                   trained_gain: np.ndarray) -> list[dict]:
    output = []
    for name in ("atom_count", "bond_count", "conjugated_bond_fraction", "ring_atom_fraction", "rwse_mean"):
        values = descriptors[name].numpy().astype(np.float64)
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
        groups = np.digitize(values, cuts[1:-1], right=True)
        for group in range(len(cuts) - 1):
            mask = groups == group
            if int(mask.sum()) < 100:
                continue
            output.append({
                "descriptor": name,
                "bin": group + 1,
                "rows": int(mask.sum()),
                "value_min": float(values[mask].min()),
                "value_max": float(values[mask].max()),
                "frozen_100k_gain_eV": float(frozen_gain[mask].mean()),
                "trained_500k_gain_eV": float(trained_gain[mask].mean()),
                "gain_change_eV": float((trained_gain[mask] - frozen_gain[mask]).mean()),
            })
    return output


def analyze(output_root: Path, matched_paths: dict[str, Path], original_paths: dict[str, Path]) -> dict:
    terminal_path = output_root / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("format") != "molgap-k1-cross-scale-frozen-diagnostic-v1" or terminal.get("complete") is not True:
        raise RuntimeError("Run terminal is absent or incomplete")
    if terminal.get("training_executed") is not False or terminal.get("model_inference_executed") is not True:
        raise RuntimeError("Run execution flags differ")
    for name in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if terminal.get(name) is not False:
            raise RuntimeError(f"Protected role flag differs: {name}")
    identity = terminal["identity"]
    if identity.get("source_commit") != SOURCE_COMMIT or identity.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Frozen audit source identity differs")
    if identity.get("manifest_sha256") != MANIFESTS or identity.get("target_transform_sha256") != TRANSFORM_SHA256:
        raise RuntimeError("Fixed data or transform identity differs")
    for arm in ARMS:
        if identity["checkpoint_sha256"].get(arm) != ARMS[arm]["model_sha256"] or identity["payload_sha256"].get(arm) != ARMS[arm]["payload_sha256"]:
            raise RuntimeError(f"Frozen {arm} checkpoint or payload identity differs")
    if len(terminal.get("chunk_sha256", {})) != 40:
        raise RuntimeError("Expected four complete 10-chunk role/arm sets")
    accepted = {
        role: {arm: load_chunks(output_root, terminal, role, arm) for arm in ARMS}
        for role in DEVELOPMENT
    }
    for arm, original_path in original_paths.items():
        if sha256(original_path) != ARMS[arm]["payload_sha256"]:
            raise RuntimeError(f"Original {arm} payload bytes differ")
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        observed = accepted["original_100k"][arm]
        if not torch.equal(observed["source_idx"], original["source_idx"].view(-1).long()) or not torch.equal(
            observed["target_eV"], original["target_eV"].view(-1).float()
        ):
            raise RuntimeError(f"Original {arm} row or target identity differs")
        max_diff = float((observed["prediction_eV"] - original["prediction_eV"].view(-1).float()).abs().max())
        if max_diff > 0.001:
            raise RuntimeError(f"Original {arm} checkpoint reproduction failed")
        if abs(float((observed["prediction_eV"] - observed["target_eV"]).abs().mean()) - ARMS[arm]["original_mae_eV"]) > 0.0001:
            raise RuntimeError(f"Original {arm} MAE reproduction failed")
    for role, arms in accepted.items():
        if not torch.equal(arms["k1"]["source_idx"], arms["pair_token"]["source_idx"]):
            raise RuntimeError(f"{role} arm source indices differ")
        if not torch.equal(arms["k1"]["target_eV"], arms["pair_token"]["target_eV"]):
            raise RuntimeError(f"{role} arm targets differ")
        for descriptor in arms["k1"]["descriptors"]:
            if not torch.equal(arms["k1"]["descriptors"][descriptor], arms["pair_token"]["descriptors"][descriptor]):
                raise RuntimeError(f"{role} arm descriptor {descriptor} differs")
    matched = {}
    unseen = accepted["unseen_500k"]
    for arm, path in matched_paths.items():
        if sha256(path) != MATCHED_PAYLOAD_SHA256[arm]:
            raise RuntimeError(f"Matched 500K {arm} payload bytes differ")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(payload["source_idx"].view(-1).long(), unseen[arm]["source_idx"]):
            raise RuntimeError(f"Matched 500K {arm} source indices differ")
        if not torch.equal(payload["target"].view(-1).float(), unseen[arm]["target_eV"]):
            raise RuntimeError(f"Matched 500K {arm} targets differ")
        matched[arm] = payload["prediction"].view(-1).float().numpy().astype(np.float64)
    target100 = accepted["original_100k"]["k1"]["target_eV"].numpy().astype(np.float64)
    target500 = unseen["k1"]["target_eV"].numpy().astype(np.float64)
    def pred(role: str, arm: str) -> np.ndarray:
        return accepted[role][arm]["prediction_eV"].numpy().astype(np.float64)
    gains = {
        "frozen_100k_on_original_100k": gain_summary(pred("original_100k", "k1"), pred("original_100k", "pair_token"), target100),
        "frozen_100k_on_unseen_500k": gain_summary(pred("unseen_500k", "k1"), pred("unseen_500k", "pair_token"), target500),
        "trained_500k_on_unseen_500k": gain_summary(matched["k1"], matched["pair_token"], target500),
    }
    frozen_gain = np.abs(pred("unseen_500k", "k1") - target500) - np.abs(pred("unseen_500k", "pair_token") - target500)
    trained_gain = np.abs(matched["k1"] - target500) - np.abs(matched["pair_token"] - target500)
    original_gain = gains["frozen_100k_on_original_100k"]["candidate_gain_eV"]
    unseen_gain = gains["frozen_100k_on_unseen_500k"]["candidate_gain_eV"]
    trained_gain_mean = gains["trained_500k_on_unseen_500k"]["candidate_gain_eV"]
    report = {
        "format": "molgap-k1-cross-scale-accepted-analysis-v1",
        "accepted": True,
        "training_executed": False,
        "model_inference_executed_in_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "source_archive_sha256": identity["source_archive_sha256"],
        "source_commit": identity["source_commit"],
        "terminal_sha256": sha256(terminal_path),
        "matched_500k_payload_sha256": MATCHED_PAYLOAD_SHA256,
        "gains": gains,
        "diagnostic_contrasts_eV": {
            "distribution_contrast_frozen_100k": unseen_gain - original_gain,
            "training_contract_contrast_on_same_500k_rows": trained_gain_mean - unseen_gain,
            "total_observed_gain_change": trained_gain_mean - original_gain,
        },
        "paired_gain_change_on_500k_rows": paired_bootstrap_mean(trained_gain - frozen_gain, n_bootstrap=10_000, seed=43),
        "strata": quintile_strata(unseen["k1"]["descriptors"], frozen_gain, trained_gain),
        "interpretation_limit": "Repeatedly used internal development roles, separately selected checkpoints, and distinct training contracts; contrasts are descriptive, not unique causal effects.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--matched-k1", type=Path, required=True)
    parser.add_argument("--matched-pair", type=Path, required=True)
    parser.add_argument("--original-k1", type=Path, required=True)
    parser.add_argument("--original-pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.output_root,
        {"k1": args.matched_k1, "pair_token": args.matched_pair},
        {"k1": args.original_k1, "pair_token": args.original_pair},
    )
    atomic_json(args.output, report)
    print(json.dumps({"accepted": True, "gains": report["gains"], "contrasts": report["diagnostic_contrasts_eV"]}))


if __name__ == "__main__":
    main()
