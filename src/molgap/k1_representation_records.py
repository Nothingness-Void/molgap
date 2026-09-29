"""Independent saved-JSON acceptance and descriptive analysis; no model imports."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from .research_memory.trace import atomic_write, file_digest, json_bytes

FLAGS = ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")
DESCRIPTORS = {"atom_count", "bond_count", "conjugated_bond_fraction", "ring_atom_fraction", "rwse_mean"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _number(value, name, lower=None, upper=None, nullable=False):
    if value is None and nullable:
        return
    _require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    _require(lower is None or value >= lower, f"{name} below bound")
    _require(upper is None or value <= upper, f"{name} above bound")


def _spectrum(s, rows, channels):
    _require(isinstance(s, dict), "Missing spectrum")
    _require(s.get("rows") == rows and s.get("channels") == channels, "Spectrum dimensions")
    ceiling = min(rows - 1, channels)
    _require(s.get("centered_rank_ceiling") == ceiling, "Spectrum rank ceiling")
    for key in ("energy_effective_rank", "stable_rank"):
        _number(s.get(key), key, 0, ceiling + 1e-6)
    _number(s.get("dispersion"), "dispersion", 0)
    for key in ("normalized_effective_rank", "leading_energy_fraction"):
        _number(s.get(key), key, 0, 1 + 1e-6)


def _row(row):
    _number(row.get("target_eV"), "target")
    _number(row.get("prediction_eV"), "prediction")
    d = row.get("descriptors", {})
    _require(set(d) == DESCRIPTORS, "Descriptor fields")
    for key, value in d.items():
        _number(value, key, 0, 1 if key.endswith("fraction") else None)
    n, e = d["atom_count"], 2 * d["bond_count"]
    _require(n >= 1 and int(n) == n and int(e) == e, "Invalid graph size")
    _require(set(row.get("layers", {})) == {"3", "6", "9"}, "Missing exchange layer")
    for layer in row["layers"].values():
        for key in ("before", "after"):
            _spectrum(layer.get(key), int(n), 192)
        before = layer["before"]["dispersion"]
        expected = layer["after"]["dispersion"] / before if before else None
        ratio = layer.get("dispersion_ratio")
        _number(ratio, "dispersion_ratio", 0, nullable=not before)
        _require(ratio is None if expected is None else math.isclose(ratio, expected, rel_tol=1e-8, abs_tol=1e-10), "Dispersion ratio inconsistent")
        _number(layer.get("update_to_hidden_norm"), "update norm", 0, nullable=True)
        _number(layer.get("update_rank1_energy_fraction"), "rank1 fraction", 0, 1+1e-6, nullable=True)
        sensitivity = layer.get("prediction_sensitivity", {})
        _require(isinstance(sensitivity, dict), "Missing sensitivity")
        _number(sensitivity.get("directional_derivative_eV"), "directional derivative")
        _number(sensitivity.get("gradient_norm"), "gradient norm", 0)
        _number(sensitivity.get("gradient_update_cosine"), "gradient cosine", -1-1e-6, 1+1e-6, nullable=True)
        if e:
            _spectrum(layer.get("bond_spectrum"), int(e), 64)
        else:
            _require(layer.get("bond_spectrum") is None, "Nonexistent bonds have spectrum")


def accept(raw: Path, plan: Path, scheduler: dict):
    """Bind actual bytes to the submitted run, not self-reported success alone."""
    raw, plan = Path(raw), Path(plan)
    receipt, inputs = _read(plan / "submission.json"), _read(plan / "input_binding.json")
    manifest = _read(raw / "completion_manifest.json")
    _require(scheduler.get("job_id") == receipt["job_id"] and scheduler.get("state") == "COMPLETED"
             and scheduler.get("exit_code") == "0:0", "Scheduler not a successful exact run")
    _number(scheduler.get("elapsed_seconds"), "scheduler elapsed", 0)
    for key in ("complete", "model_inference_executed"):
        _require(manifest.get(key) is True, f"Missing {key}")
    _require(manifest.get("training_executed") is False, "Training was not forbidden")
    for key in FLAGS:
        _require(manifest.get(key) is False, f"Protected role: {key}")
    _require(manifest.get("experiment_purpose") == "NO_TRAIN" and manifest.get("comparison_class") == "CONTEXT_ONLY", "Claim mismatch")
    _require(manifest.get("physical_batch") == 128 and manifest.get("precision") == "fp32", "Runtime contract")
    _require(manifest.get("panel_rows_per_scale") == 1024, "Panel size")
    for key in ("source_commit", "source_archive_sha256"):
        _require(manifest.get(key) == receipt[key], f"Wrong {key}")
    # The reference pointer is resolved by the release gate, not accepted from raw outputs.
    repo = plan.resolve().parents[2]
    reference = _read(repo / inputs["model100_reference"])
    _require(manifest.get("checkpoint_sha256") == {"100": reference["checkpoint_identity"], "500": inputs["model500_sha256"]}, "Checkpoint mismatch")
    expected_files = {f"scale{s}_chunk{c:02d}.json" for s in (100, 500) for c in range(8)} | {
        "row_manifest.json", "started.json", "role_events.json", "reproduction.json"}
    hashes = manifest.get("artifact_sha256", {})
    _require(set(hashes) == expected_files, "Missing/unexpected artifact or unsafe path")
    for name, digest in hashes.items():
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid digest")
        path = raw / name
        _require(path.is_file() and not path.is_symlink(), "Missing/nonregular artifact")
        _require(file_digest(path) == digest, f"Hash mismatch: {name}")
    panel = _read(plan / "row_manifest.json")
    _require(file_digest(plan / "row_manifest.json") == inputs["row_manifest_sha256"], "Frozen panel changed")
    _require(_read(raw / "row_manifest.json") == panel, "Remote panel changed")
    ids = panel["source_idx"]
    _require(len(ids) == 1024 and ids == sorted(set(ids)) and all(type(i) is int and 500000 <= i < 550000 for i in ids), "Invalid panel IDs")
    started = _read(raw / "started.json")
    _require(started.get("job_id") == receipt["job_id"], "Wrong started job")
    for key in ("source_commit", "source_archive_sha256", "training_executed", "model_inference_executed", *FLAGS):
        _require(started.get(key) == manifest[key], f"Started/terminal mismatch: {key}")
    repro = _read(raw / "reproduction.json")
    _require(set(repro) == {"100", "500"}, "Incomplete reproduction")
    for record in repro.values():
        _number(record.get("max_abs_eV"), "reproduction maximum", 0, .001)
        _number(record.get("mae_difference_eV"), "reproduction MAE", 0, .0001)
    expected_events = []
    for scale in (100, 500):
        expected_events.extend([
            {"scale": scale, "role": "original_100k" if scale == 100 else "unseen_500k", "purpose": "prediction-reproduction", "rows": 50000},
            {"scale": scale, "role": "unseen_500k", "purpose": "representation-panel", "rows": 1024},
        ])
    _require(_read(raw / "role_events.json") == expected_events, "Role event mismatch")
    rows = {}
    for scale in (100, 500):
        rows[scale] = []
        for chunk in range(8):
            values = _read(raw / f"scale{scale}_chunk{chunk:02d}.json")
            _require(isinstance(values, list) and len(values) == 128, "Chunk size")
            _require([r.get("source_idx") for r in values] == ids[chunk*128:(chunk+1)*128], "Chunk row order")
            for row in values:
                _row(row)
            rows[scale].extend(values)
    for a, b in zip(rows[100], rows[500]):
        _require(a["target_eV"] == b["target_eV"] and a["descriptors"] == b["descriptors"], "Paired targets/descriptors differ")
    for key in ("wall_seconds", "device_hours", "peak_allocated_bytes"):
        _number(manifest.get(key), key, 0)
    _require(math.isclose(manifest["device_hours"] * 3600, manifest["wall_seconds"], abs_tol=1e-6), "Cost units inconsistent")
    return {"accepted": True, "model_inference_executed_by_acceptance": False,
        "job_id": receipt["job_id"], "comparison_class": "CONTEXT_ONLY", "training_replay_ready": False,
        "completion_sha256": file_digest(raw / "completion_manifest.json"), "panel_rows": len(ids),
        "scheduler": scheduler, "runtime_gates": "source-bound completion implies hook, coupling and immutable-state assertions passed; raw per-batch maxima unavailable",
        "cost_caveat": "native process time differs from allocated Slurm time; no cross-hardware conversion"}, rows


def _summary(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values))}


def _adjusted_correlation(x, y, atoms):
    valid = [i for i, v in enumerate(x) if v is not None]
    if len(valid) < 10:
        return None
    x, y, atoms = (np.asarray(v, dtype=float)[valid] for v in (x, y, atoms))
    # Prespecified linear atom-count adjustment; not full chemistry control.
    design = np.column_stack([np.ones(len(atoms)), atoms])
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(rows):
    a, b = rows[100], rows[500]
    errors = {s: np.asarray([abs(r["prediction_eV"]-r["target_eV"]) for r in rows[s]]) for s in rows}
    atoms = [r["descriptors"]["atom_count"] for r in a]
    groups = {"all": list(range(len(a))), "small_sparse_low_conjugation": [], "high_ring": [], "high_rwse": [], "middle_control": []}
    for i, row in enumerate(a):
        d = row["descriptors"]
        small = d["atom_count"] <= 12 and d["bond_count"] <= 12 and d["conjugated_bond_fraction"] <= .27272727
        ring, rwse = d["ring_atom_fraction"] > .75, d["rwse_mean"] > .144207
        for key, flag in (("small_sparse_low_conjugation",small),("high_ring",ring),("high_rwse",rwse),("middle_control",not (small or ring or rwse))):
            if flag:
                groups[key].append(i)
    result = {"interpretation": "Descriptive endpoint diagnostic; scales differ in training exposure and selection. No causal architecture or promotion claim.",
        "panel_mae_eV": {str(s):float(e.mean()) for s,e in errors.items()}, "groups": {}, "layers": {}}
    for name, idx in groups.items():
        result["groups"][name] = {"n":len(idx), "error_delta_500_minus_100":_summary((errors[500]-errors[100])[idx])}
    for layer in ("3","6","9"):
        layer_result = {}
        for name, getter in {
            "normalized_rank_after": lambda r:r["after"]["normalized_effective_rank"],
            "rank_change_across_exchange": lambda r:r["after"]["normalized_effective_rank"]-r["before"]["normalized_effective_rank"],
            "dispersion_ratio":lambda r:r["dispersion_ratio"],
            "gradient_update_cosine":lambda r:r["prediction_sensitivity"]["gradient_update_cosine"],
            "absolute_directional_derivative":lambda r:abs(r["prediction_sensitivity"]["directional_derivative_eV"]),
        }.items():
            values = {s:[getter(r["layers"][layer]) for r in rows[s]] for s in (100,500)}
            delta = [v-u if u is not None and v is not None else None for u,v in zip(values[100],values[500])]
            layer_result[name] = {"by_scale":{str(s):_summary(values[s]) for s in values},
                "paired_delta":_summary(delta), "group_delta":{g:_summary([delta[i] for i in idx]) for g,idx in groups.items()},
                "atom_count_adjusted_correlation_with_error_delta":_adjusted_correlation(delta, errors[500]-errors[100], atoms)}
        result["layers"][layer] = layer_result
    return result


def write_reports(raw, plan, scheduler_path, output):
    output = Path(output)
    _require(not output.exists(), "Use a new output directory; never overwrite evidence")
    accepted, rows = accept(Path(raw), Path(plan), _read(scheduler_path))
    analysis = analyze(rows)
    output.mkdir(parents=True)
    atomic_write(output / "acceptance.json", json_bytes(accepted))
    atomic_write(output / "analysis.json", json_bytes(analysis))
    atomic_write(output / "REVIEW_REQUIRED.md", (
        "# Controller review required\n\nMechanical acceptance passed. Review analysis.json; do not infer a\n"
        "mechanism from rank or correlation alone. Record scale/exposure confounds,\n"
        "then use the existing NO_TRAIN terminal pipeline. No training is released.\n"
    ).encode())
