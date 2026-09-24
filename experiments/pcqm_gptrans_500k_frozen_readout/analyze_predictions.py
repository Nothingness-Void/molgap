"""Read-only, row-aligned attribution of retained matched-500K predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molgap.training_reproducibility import atomic_json, sha256_file


ARMS = {
    "baseline": "pcqm_500k_v4_stage5/gptrans-v7/evidence/gptrans",
    "pair_norm": "pcqm_gptrans_pair_norm_500k_s42_v1/gptrans_pair_update_norm",
    "noisy_nodes": "pcqm_gptrans_noisy_nodes_500k_s42_v1/pcqm_gptrans_noisy_nodes_500k",
    "joint": "pcqm_gptrans_noisy_pair_norm_500k_s42_v1/pcqm_gptrans_noisy_pair_norm_500k",
}
MANIFEST_SHA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
DEV_SHARD_SHA = "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1"
TRACE_PATHS = {
    "baseline": "pcqm_500k_v4_evidence",
    "pair_norm": "pcqm_gptrans_pair_norm_500k",
    "noisy_nodes": "pcqm_gptrans_noisy_nodes_500k",
    "joint": "pcqm_gptrans_noisy_pair_norm_500k",
}


def _bins(values: np.ndarray, improvement: np.ndarray) -> list[dict]:
    edges = np.quantile(values, np.linspace(0, 1, 6))
    result = []
    for index in range(5):
        selected = (values >= edges[index]) & (
            values < edges[index + 1] if index < 4 else values <= edges[index + 1]
        )
        result.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "rows": int(selected.sum()),
            "mean_gain_eV": float(improvement[selected].mean()),
            "row_win_fraction": float(np.mean(improvement[selected] > 0)),
        })
    return result


def analyze(records_root: Path, cache_root: Path, repo_root: Path) -> dict:
    if sha256_file(cache_root / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("Fixed 500K graph manifest changed")
    dev_shard = cache_root / "train/train_shard_0010.pt"
    if sha256_file(dev_shard) != DEV_SHARD_SHA:
        raise RuntimeError("Fixed 500K development shard changed")
    _, slices = torch.load(dev_shard, map_location="cpu", weights_only=False)
    atoms = (slices["x"][1:] - slices["x"][:-1]).numpy().astype(np.float64)

    payloads = {}
    identities = {}
    for arm, relative in ARMS.items():
        folder = records_root / relative
        if arm == "baseline":
            manifest_sha = "7cf5243d0a5a34818f8a86801347be9dffb6fd6b81a66df34958c9582a79c536"
            prediction_sha = "0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf"
            best_epoch = 56
        else:
            manifest_path = folder / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["status"] != "COMPLETE":
                raise RuntimeError(f"Incomplete selected arm: {arm}")
            manifest_sha = sha256_file(manifest_path)
            prediction_sha = manifest["artifacts"]["best_predictions.pt"]
            best_epoch = manifest["best_epoch"]
        path = folder / "best_predictions.pt"
        digest = sha256_file(path)
        if digest != prediction_sha:
            raise RuntimeError(f"Prediction hash mismatch: {arm}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(payload["source_idx"], torch.arange(500000, 550000)):
            raise RuntimeError(f"Wrong development rows: {arm}")
        if not all(torch.isfinite(payload[key]).all() for key in ("prediction", "target")):
            raise RuntimeError(f"Nonfinite prediction or target: {arm}")
        payloads[arm] = payload
        identities[arm] = {
            "manifest_sha256": manifest_sha,
            "predictions_sha256": digest,
            "best_epoch": best_epoch,
        }
    baseline = payloads["baseline"]
    target = baseline["target"].double().numpy()
    if atoms.shape[0] != target.shape[0]:
        raise RuntimeError("Atom counts and prediction rows differ")
    for arm, payload in payloads.items():
        if not torch.equal(payload["target"], baseline["target"]):
            raise RuntimeError(f"Target mismatch: {arm}")

    baseline_error = (baseline["prediction"] - baseline["target"]).abs().double().numpy()
    rng = np.random.default_rng(20260925)
    results = {}
    for arm, payload in payloads.items():
        error = (payload["prediction"] - payload["target"]).abs().double().numpy()
        gain = baseline_error - error
        draws = np.empty(2000, dtype=np.float64)
        for start in range(0, len(draws), 100):
            indices = rng.integers(0, len(gain), size=(100, len(gain)))
            draws[start:start + 100] = gain[indices].mean(axis=1)
        results[arm] = {
            "mae_eV": float(error.mean()),
            "gain_vs_baseline_eV": float(gain.mean()),
            "row_win_fraction": float(np.mean(gain > 0)),
            "row_bootstrap_95pct_gain_eV": np.quantile(draws, [0.025, 0.975]).tolist(),
            "gain_by_target_quintile": _bins(target, gain),
            "gain_by_atom_count_quintile": _bins(atoms, gain),
        }
    traces = {}
    for arm, directory in TRACE_PATHS.items():
        path = repo_root / "experiments" / directory / "results" / "canonical_trace.json"
        observations = json.loads(path.read_text(encoding="utf-8"))["observations"]
        if len(observations) != 60:
            raise RuntimeError(f"Expected 60 matched 500K trace observations: {arm}")
        traces[arm] = observations
    matched = {}
    for arm in ("pair_norm", "noisy_nodes", "joint"):
        rows = []
        for epoch in (9, 19, 29, 39, 49, 59):
            reference = traces["baseline"][epoch]
            candidate = traces[arm][epoch]
            if (reference["optimizer_step"] != candidate["optimizer_step"] or
                reference["sample_presentations"] != candidate["sample_presentations"] or
                reference["epoch_or_pass"] != candidate["epoch_or_pass"]):
                raise RuntimeError(f"Matched 500K trace cursor differs: {arm}/{epoch}")
            rows.append({
                "epoch_zero_based": epoch,
                "optimizer_step": reference["optimizer_step"],
                "sample_presentations": reference["sample_presentations"],
                "live_development_gain_eV": reference["live_dev_metric"] - candidate["live_dev_metric"],
                "online_training_gain_eV": reference["live_train_metric"] - candidate["live_train_metric"],
            })
        matched[arm] = rows
    return {
        "format": "molgap-gptrans-500k-prediction-attribution-v1",
        "scope": "reused_internal_development_only_not_promotion",
        "rows": len(target),
        "source_idx_start": 500000,
        "source_idx_stop": 550000,
        "graph_manifest_sha256": MANIFEST_SHA,
        "development_shard_sha256": DEV_SHARD_SHA,
        "prediction_identities": identities,
        "results": results,
        "matched_trace_observations": matched,
        "trace_caveat": "online training metrics and development evaluation are not interchangeable; matched epochs do not isolate data scale or optimization",
        "uncertainty_scope": "row resampling only; no training-seed or selection uncertainty",
        "protected_roles_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.records_root, args.cache_root, args.repo_root)
    atomic_json(args.output, result)
    print(json.dumps({arm: data["gain_vs_baseline_eV"] for arm, data in result["results"].items()}))
