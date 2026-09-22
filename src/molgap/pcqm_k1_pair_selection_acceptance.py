"""No-inference acceptance for the frozen PairToken selection diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pcqm_fixed_datasets import sha256_file


EXPECTED_INPUTS = {
    "checkpoint_sha256": "ac8b576a3c7e44d50012a885e13e4c5904ad17c3574319796b13518efe98d1cc",
    "pair_payload_sha256": "b67be3f9ba0c67ca0eb1cf7dffa00ad535e9f04f5f8371aadad1e6a2711e0331",
    "k1_payload_sha256": "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91",
    "cache_manifest_sha256": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
}
METRICS = (
    "top20_mass", "effective_pair_fraction", "diagonal_mass",
    "bonded_mass", "two_hop_mass", "nonlocal_mass", "atom_count",
    "conjugated_bond_fraction", "k1_absolute_error", "paired_gain_eV",
)


def accept(output_root: Path) -> dict:
    diagnostic_path = output_root / "diagnostic.json"
    result = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    progress = json.loads((output_root / "progress.json").read_text(encoding="utf-8"))
    if result.get("format") != "molgap-k1-pair-selection-diagnostic-v1":
        raise RuntimeError("Unexpected diagnostic format")
    if result.get("complete") is not True or result.get("development_rows") != 50_000:
        raise RuntimeError("Diagnostic incomplete")
    if result.get("training_executed") is not False or result.get("model_inference_executed") is not True:
        raise RuntimeError("Execution mode mismatch")
    for role in ("official_validation", "test_dev", "test_challenge"):
        if result.get(f"{role}_role_read") is not False:
            raise RuntimeError(f"Protected role read: {role}")
    if result.get("promotion_authorized") is not False:
        raise RuntimeError("Diagnostic cannot authorize promotion")
    for key, expected in EXPECTED_INPUTS.items():
        if result.get(key) != expected or progress.get("input_identity", {}).get(key) != expected:
            raise RuntimeError(f"Input identity mismatch: {key}")
    rows = progress.get("chunks", [])
    if len(rows) != 10 or [item["start"] for item in rows] != list(range(0, 50_000, 5_000)):
        raise RuntimeError("Chunk coverage mismatch")
    arrays = {name: [] for name in METRICS}
    for row in rows:
        start, end = row["start"], row["end"]
        if end != start + 5_000:
            raise RuntimeError("Chunk length mismatch")
        path = output_root / f"rows_{start:05d}_{end:05d}.npz"
        digest = sha256_file(path)
        if digest != row["sha256"] or result["chunk_sha256"].get(str(start)) != digest:
            raise RuntimeError(f"Chunk hash mismatch: {start}")
        if row["prediction_max_absolute_difference_eV"] > 5e-4:
            raise RuntimeError("Frozen prediction reproduction failed")
        if row["mae_absolute_difference_eV"] > 5e-6:
            raise RuntimeError("Frozen MAE reproduction failed")
        with np.load(path) as chunk:
            if set(chunk.files) != set(METRICS):
                raise RuntimeError("Chunk field mismatch")
            for name in METRICS:
                values = chunk[name]
                if values.shape != (5_000,) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid chunk metric: {name}")
                arrays[name].append(values)
    arrays = {name: np.concatenate(parts).astype(np.float64)
              for name, parts in arrays.items()}
    top = arrays["top20_mass"]
    hard = arrays["k1_absolute_error"] >= np.quantile(arrays["k1_absolute_error"], 0.9)
    non_small = arrays["atom_count"] > 12
    if not bool(non_small.any()) or not bool(hard.any()):
        raise RuntimeError("Declared strata absent")
    gate = {
        "median_top20_at_least_0_80": bool(np.median(top) >= 0.8),
        "fraction_at_least_0_80_ge_0_70": bool(np.mean(top >= 0.8) >= 0.7),
        "more_than_12_atoms_median_at_least_0_80": bool(np.median(top[non_small]) >= 0.8),
        "hardest_k1_decile_median_at_least_0_80": bool(np.median(top[hard]) >= 0.8),
    }
    if result["gate"] != gate or result["candidate_question_qualified"] != all(gate.values()):
        raise RuntimeError("Reported gate differs from recomputation")
    for name in ("top20_mass", "effective_pair_fraction", "diagonal_mass",
                 "bonded_mass", "two_hop_mass", "nonlocal_mass", "paired_gain_eV"):
        reported = result["strata"]["all"][name]
        for field, value in (("mean", float(np.mean(arrays[name]))),
                             ("median", float(np.median(arrays[name])))):
            if abs(reported[field] - value) > 1e-8:
                raise RuntimeError(f"Aggregate mismatch: {name}/{field}")
    if result["strata"]["all"]["rows"] != 50_000:
        raise RuntimeError("Aggregate row count mismatch")
    return {
        "format": "molgap-k1-pair-selection-acceptance-v1",
        "accepted": True,
        "model_inference_executed_by_acceptance": False,
        "training_executed_by_acceptance": False,
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "chunks": len(rows),
        "development_rows": 50_000,
        "candidate_question_qualified": all(gate.values()),
        "top20_mass_median": float(np.median(top)),
        "top20_mass_mean": float(np.mean(top)),
        "top20_mass_fraction_ge_0_80": float(np.mean(top >= 0.8)),
        "effective_pair_fraction_median": float(np.median(arrays["effective_pair_fraction"])),
        "nonlocal_mass_mean": float(np.mean(arrays["nonlocal_mass"])),
        "paired_gain_mean_eV": float(np.mean(arrays["paired_gain_eV"])),
    }
