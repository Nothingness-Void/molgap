"""Screen a global 2D/geometry convex mix from the accepted prediction pair."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.pcqm_geometry_component_attribution.analyze import (
    BASELINE,
    CANDIDATE,
    EVIDENCE_ID,
    PREDICTION,
    _load_prediction,
    _read_json,
)
from molgap.pcqm_distance_angle_scale import sha256_file


GRID = np.arange(21, dtype=np.float64) / 20.0
EXPECTED_BASELINE = 0.1176588833
EXPECTED_GEOMETRY = 0.1141408384


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def _mixture(base: np.ndarray, geometry: np.ndarray, weight: float) -> np.ndarray:
    return (1.0 - weight) * base + weight * geometry


def screen(repo: Path, artifact_root: Path) -> dict:
    started = time.perf_counter()
    source = repo / "experiments/pcqm_distance_angle_500k"
    envelope = _read_json(source / "v5_evidence.json")
    if envelope["evidence_id"] != EVIDENCE_ID:
        raise ValueError("source evidence identity changed")
    accepted = _read_json(source / "results/local_acceptance_20260923.json")
    if accepted.get("accepted") is not True:
        raise ValueError("source pair acceptance is not true")
    hashes = {item["locator"].replace("\\", "/"): item["sha256"] for item in envelope["artifacts"]}
    payloads = {}
    inputs = {}
    for arm in (BASELINE, CANDIDATE):
        locator = (
            "platforms/_records/scnet/pcqm_distance_angle_500k_20260923/"
            f"remote/output/{arm}/{PREDICTION}"
        )
        digest = hashes[locator]
        payloads[arm] = _load_prediction(artifact_root / locator, digest)
        inputs[arm] = {"locator": locator, "sha256": digest}
    base_rows, geo_rows = payloads[BASELINE], payloads[CANDIDATE]
    if not np.array_equal(base_rows["source_idx"], geo_rows["source_idx"]):
        raise ValueError("source indices differ")
    if not np.array_equal(base_rows["target_eV"], geo_rows["target_eV"]):
        raise ValueError("targets differ")
    index = base_rows["source_idx"].astype(np.int64)
    target = base_rows["target_eV"].astype(np.float64)
    base = base_rows["prediction_eV"].astype(np.float64)
    geometry = geo_rows["prediction_eV"].astype(np.float64)
    baseline_mae, geometry_mae = _mae(base, target), _mae(geometry, target)
    if not math.isclose(baseline_mae, EXPECTED_BASELINE, abs_tol=5e-8):
        raise ValueError("baseline MAE differs from accepted result")
    if not math.isclose(geometry_mae, EXPECTED_GEOMETRY, abs_tol=5e-8):
        raise ValueError("geometry MAE differs from accepted result")
    oof = np.empty_like(target)
    folds = []
    for fold in range(5):
        held = index % 5 == fold
        train = ~held
        scores = [_mae(_mixture(base[train], geometry[train], weight), target[train]) for weight in GRID]
        minimum = min(scores)
        selected = max(i for i, score in enumerate(scores) if math.isclose(score, minimum, abs_tol=1e-12))
        weight = float(GRID[selected])
        oof[held] = _mixture(base[held], geometry[held], weight)
        folds.append({
            "fold": fold,
            "rows": int(held.sum()),
            "geometry_weight": weight,
            "training_mae_eV": scores[selected],
            "oof_mae_eV": _mae(oof[held], target[held]),
            "geometry_mae_eV": _mae(geometry[held], target[held]),
        })
    for row in folds:
        row["oof_gain_over_geometry_eV"] = row["geometry_mae_eV"] - row["oof_mae_eV"]
    tails = {}
    for name, low, high in (("2_4", 2.0, 4.0), ("8_inf", 8.0, math.inf)):
        mask = (target >= low) & (target < high)
        tails[name] = {
            "rows": int(mask.sum()),
            "geometry_mae_eV": _mae(geometry[mask], target[mask]),
            "oof_mae_eV": _mae(oof[mask], target[mask]),
        }
    oof_mae = _mae(oof, target)
    gain = geometry_mae - oof_mae
    screen_pass = (
        gain >= 0.001
        and sum(row["oof_gain_over_geometry_eV"] > 0 for row in folds) >= 4
        and all(item["oof_mae_eV"] <= item["geometry_mae_eV"] for item in tails.values())
    )
    return {
        "format": "molgap-pcqm-geometry-fusion-screen-v1",
        "source_evidence_id": EVIDENCE_ID,
        "source_inputs": inputs,
        "source_envelope_sha256": sha256_file(source / "v5_evidence.json"),
        "source_acceptance_sha256": sha256_file(source / "results/local_acceptance_20260923.json"),
        "aligned_rows": len(target),
        "source_idx_range": [int(index.min()), int(index.max())],
        "baseline_mae_eV": baseline_mae,
        "geometry_mae_eV": geometry_mae,
        "fixed_weight_mae_eV": {str(weight): _mae(_mixture(base, geometry, weight), target) for weight in (0.5, 0.75, 1.0)},
        "oof_mae_eV": oof_mae,
        "oof_gain_over_geometry_eV": gain,
        "folds": folds,
        "target_gap_tail_descriptions": tails,
        "frozen_screen_pass": screen_pass,
        "role_use": {
            "internal_development": "previously_consumed_reused_for_exploratory_weight_selection",
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "model_training_executed": False,
        "model_inference_executed": False,
        "local_wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(args.repo_root.resolve(), args.artifact_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"oof_gain_over_geometry_eV": result["oof_gain_over_geometry_eV"], "frozen_screen_pass": result["frozen_screen_pass"]}))


if __name__ == "__main__":
    main()
