"""Review accepted fixed-weight blends using retained predictions only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from molgap.training_reproducibility import atomic_json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    helper_path = root / "experiments/pcqm_500k_v4_evidence/analyze_local_ablation.py"
    spec = importlib.util.spec_from_file_location("matched_v4_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    references = root / "platforms/_records/kaggle/training/pcqm_500k_v4_evidence_reference_predictions"
    candidates = root / "platforms/_records/kaggle/training/geometry_v4_500k_continuation_attempt_003/output/evidence"
    inputs = {
        "gptrans_2d": (references / "gptrans__best_predictions.pt", "0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf"),
        "k1_2d": (references / "neural_atom_k1__best_predictions.pt", "68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b"),
        "gptrans_distance": (candidates / "gptrans_distance_only/best_predictions.pt", "85488eb225721274e37097554a9ceb81371267d5b404e939bcfb96403d982105"),
        "k1_geometry": (candidates / "k1_distance_angle/best_predictions.pt", "a5363aaf2654996e169cf62277db68e42221a08102d94b905bec9aeb66ff0a23"),
    }
    # Bind the final K1 bytes to the already accepted immutable manifest.
    manifest = json.loads((candidates / "k1_distance_angle/stage_manifest.json").read_text())
    if inputs["k1_geometry"][1] != manifest["artifacts"]["best_predictions.pt"]:
        raise ValueError("Accepted K1 manifest binding mismatch")
    payloads = {name: helper._load(path) for name, (path, _) in inputs.items()}
    anchor = payloads["gptrans_2d"]
    for name, payload in payloads.items():
        if payload["sha256"] != inputs[name][1]:
            raise ValueError(f"Accepted artifact hash mismatch: {name}")
        if not np.array_equal(payload["source_idx"], np.arange(500000, 550000)):
            raise ValueError(f"Row identity mismatch: {name}")
        if not np.array_equal(payload["target"], anchor["target"]):
            raise ValueError(f"Target identity mismatch: {name}")
    y = anchor["target"]
    predictions = {name: payload["prediction"] for name, payload in payloads.items()}
    blends = {
        "original_2d_fixed_50_50": (predictions["gptrans_2d"] + predictions["k1_2d"]) / 2,
        "geometry_fixed_50_50": (predictions["gptrans_distance"] + predictions["k1_geometry"]) / 2,
        "distance_with_original_k1_fixed_50_50": (predictions["gptrans_distance"] + predictions["k1_2d"]) / 2,
        "original_gptrans_with_geometry_k1_fixed_50_50": (predictions["gptrans_2d"] + predictions["k1_geometry"]) / 2,
    }
    predictions.update(blends)
    errors = {name: np.abs(pred - y) for name, pred in predictions.items()}
    comparisons = [
        ("geometry_fixed_50_50", "original_2d_fixed_50_50"),
        ("gptrans_distance", "gptrans_2d"),
        ("k1_geometry", "k1_2d"),
        ("distance_with_original_k1_fixed_50_50", "original_2d_fixed_50_50"),
        ("geometry_fixed_50_50", "distance_with_original_k1_fixed_50_50"),
    ]
    paired = {}
    for candidate, reference in comparisons:
        difference = errors[reference] - errors[candidate]
        report = helper._bootstrap(difference)
        report["candidate_lower_row_error_fraction"] = float((difference > 0).mean())
        paired[f"{candidate}__over__{reference}"] = report
    result = {
        "format": "molgap-fixed-blend-review-v1",
        "n": 50000,
        "source_idx_and_targets_exactly_aligned": True,
        "model_inference_executed": False,
        "training_executed": False,
        "new_remote_job_submitted": False,
        "analysis_scope": "supplementary existing-question scale review; same already used development role; hybrid decompositions post-hoc and non-promotional",
        "bootstrap_seed": helper.BOOTSTRAP_SEED,
        "bootstrap_replicates": helper.BOOTSTRAP_REPLICATES,
        "bootstrap_semantics": "paired development-row resampling; positive mean is improvement; not training-seed uncertainty",
        "artifact_sha256": {name: payload["sha256"] for name, payload in payloads.items()},
        "artifact_paths": {name: path.relative_to(root).as_posix() for name, (path, _) in inputs.items()},
        "mae_eV": {name: float(error.mean()) for name, error in errors.items()},
        "paired_improvements": paired,
        "materiality_context_eV": 0.003,
        "materiality_context": "existing individual-arm promotion floor; not a retrospectively preregistered blend-versus-blend gate",
    }
    output = Path(__file__).with_name("fixed_blend_review.json")
    atomic_json(output, result)
    print(json.dumps({"mae_eV": result["mae_eV"], "paired_improvements": paired}, indent=2))


if __name__ == "__main__":
    main()
