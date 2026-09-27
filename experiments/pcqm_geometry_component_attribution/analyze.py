"""Describe the accepted distance-plus-angle pair without model execution."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_distance_angle_scale import sha256_file


BASELINE = "ogb_edge_state_structural_gps9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_gps9"
PREDICTION = "direct_gap_development.pt"
EVIDENCE_ID = "pcqm-distance-angle-triangle-500k-20260911"
EXPECTED_GAIN = -0.003518044948577881


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prediction(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"accepted prediction hash mismatch or file absent: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("official_validation_role_read") is not False or payload.get("test_dev_role_read") is not False:
        raise ValueError("protected role flag changed")
    out = {}
    for key in ("source_idx", "target_eV", "prediction_eV"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != 50_000:
            raise ValueError(f"invalid {key} shape")
        out[key] = value.numpy()
    if not np.array_equal(out["source_idx"], np.arange(500_000, 550_000)):
        raise ValueError("development source rows differ from accepted role")
    if not np.isfinite(out["target_eV"]).all() or not np.isfinite(out["prediction_eV"]).all():
        raise ValueError("nonfinite target or prediction")
    return out


def _metrics(mask: np.ndarray, baseline_error: np.ndarray, candidate_error: np.ndarray) -> dict:
    base = baseline_error[mask]
    candidate = candidate_error[mask]
    if not len(base):
        return {"rows": 0, "candidate_minus_baseline_mae_eV": None}
    delta = candidate - base
    return {
        "rows": int(len(base)),
        "baseline_mae_eV": float(base.mean()),
        "candidate_mae_eV": float(candidate.mean()),
        "candidate_minus_baseline_mae_eV": float(delta.mean()),
        "candidate_win_rate": float(np.mean(delta < 0)),
        "candidate_tie_rate": float(np.mean(delta == 0)),
    }


def _quartiles(values: np.ndarray, baseline_error: np.ndarray, candidate_error: np.ndarray) -> dict:
    # Stable rank bins keep each descriptive quartile at exactly 12,500 rows.
    order = np.argsort(values, kind="stable")
    output = {}
    for index, selected in enumerate(np.array_split(order, 4), 1):
        mask = np.zeros(len(values), dtype=bool)
        mask[selected] = True
        output[f"Q{index}"] = {
            "value_min": float(values[selected].min()),
            "value_max": float(values[selected].max()),
            **_metrics(mask, baseline_error, candidate_error),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo_root.resolve()
    source = repo / "experiments/pcqm_distance_angle_500k"
    envelope = _read_json(source / "v5_evidence.json")
    if envelope["evidence_id"] != EVIDENCE_ID:
        raise ValueError("source envelope identity changed")
    accepted = _read_json(source / "results/local_acceptance_20260923.json")
    if accepted.get("accepted") is not True:
        raise ValueError("owning pair was not accepted")
    artifacts = {a["locator"].replace("\\", "/"): a["sha256"] for a in envelope["artifacts"]}
    rows = {}
    inputs = {}
    for arm in (BASELINE, CANDIDATE):
        locator = (
            "platforms/_records/scnet/pcqm_distance_angle_500k_20260923/"
            f"remote/output/{arm}/{PREDICTION}"
        )
        expected_sha = artifacts[locator]
        rows[arm] = _load_prediction(args.artifact_root / locator, expected_sha)
        inputs[arm] = {"locator": locator, "sha256": expected_sha}
    baseline, candidate = rows[BASELINE], rows[CANDIDATE]
    if not np.array_equal(baseline["source_idx"], candidate["source_idx"]):
        raise ValueError("source rows differ between arms")
    if not np.array_equal(baseline["target_eV"], candidate["target_eV"]):
        raise ValueError("targets differ between arms")
    target = baseline["target_eV"].astype(np.float64)
    pred_base = baseline["prediction_eV"].astype(np.float64)
    pred_candidate = candidate["prediction_eV"].astype(np.float64)
    base_error = np.abs(pred_base - target)
    candidate_error = np.abs(pred_candidate - target)
    all_rows = np.ones(len(target), dtype=bool)
    overall = _metrics(all_rows, base_error, candidate_error)
    if not math.isclose(
        overall["candidate_minus_baseline_mae_eV"], EXPECTED_GAIN, abs_tol=5e-8
    ):
        raise ValueError("recomputed paired gain differs from accepted result")
    if not math.isclose(
        overall["baseline_mae_eV"], accepted["local_no_inference_recomputed"]["arms"][BASELINE]["best_development_mae_eV"], abs_tol=5e-8
    ) or not math.isclose(
        overall["candidate_mae_eV"], accepted["local_no_inference_recomputed"]["arms"][CANDIDATE]["best_development_mae_eV"], abs_tol=5e-8
    ):
        raise ValueError("recomputed arm MAE differs from accepted result")
    gap_bins = {}
    for label, lower, upper in (("0_2", 0, 2), ("2_4", 2, 4), ("4_6", 4, 6), ("6_8", 6, 8), ("8_inf", 8, math.inf)):
        gap_bins[label] = _metrics((target >= lower) & (target < upper), base_error, candidate_error)
    if sum(bin_["rows"] for bin_ in gap_bins.values()) != len(target):
        raise ValueError("target-gap bins do not cover development rows")
    trace_fields = ("optimizer_step", "sample_presentations", "checkpoint_identity", "device_time_seconds", "cumulative_wall_time_seconds")
    trace_audit = {}
    for arm, trace_name in ((BASELINE, "baseline_canonical_trace.json"), (CANDIDATE, "candidate_canonical_trace.json")):
        path = source / "results" / trace_name
        trace = _read_json(path)
        observations = trace["observations"]
        trace_audit[arm] = {
            "locator": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
            "observations": len(observations),
            "non_null_counts": {field: sum(row.get(field) is not None for row in observations) for field in trace_fields},
        }
    reuse_index = _read_json(repo / "research_memory/derived/reference_reuse_index.json")
    reference = next(ref for ref in reuse_index["references"] if ref["reference_id"] == EVIDENCE_ID)
    report = {
        "format": "molgap-pcqm-geometry-component-attribution-diagnostic-v1",
        "source_evidence_id": EVIDENCE_ID,
        "source_inputs": inputs,
        "source_envelope_sha256": sha256_file(source / "v5_evidence.json"),
        "source_acceptance_sha256": sha256_file(source / "results/local_acceptance_20260923.json"),
        "aligned_rows": len(target),
        "source_idx_range": [int(baseline["source_idx"].min()), int(baseline["source_idx"].max())],
        "target_identity_equal": True,
        "all_predictions_finite": True,
        "overall": {
            **overall,
            "baseline_signed_bias_eV": float(np.mean(pred_base - target)),
            "candidate_signed_bias_eV": float(np.mean(pred_candidate - target)),
            "signed_residual_correlation": float(np.corrcoef(pred_base - target, pred_candidate - target)[0, 1]),
            "absolute_residual_correlation": float(np.corrcoef(base_error, candidate_error)[0, 1]),
        },
        "strata": {
            "target_gap_eV": gap_bins,
            "baseline_abs_error_quartile": _quartiles(base_error, base_error, candidate_error),
            "prediction_disagreement_quartile": _quartiles(np.abs(pred_candidate - pred_base), base_error, candidate_error),
        },
        "reference_qualification": {
            "comparison_status": reference["comparison_status"],
            "transfer_status": reference["transfer_status"],
            "historical_trace": trace_audit,
        },
        "role_use": {
            "internal_development": "previously_consumed_reused_for_exploratory_analysis",
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "model_training_executed": False,
        "model_inference_executed": False,
        "local_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "reference_qualification": report["reference_qualification"]}, indent=2))


if __name__ == "__main__":
    main()
