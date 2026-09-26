"""Saved-prediction and native-telemetry analysis; no model or graph execution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .research_memory.trace import file_digest, load_canonical_trace
from .router import paired_bootstrap_mean


def paired_saved_errors(reference: dict, candidate: dict) -> dict:
    for key in ("source_idx", "target_eV"):
        if not torch.equal(reference[key].view(-1), candidate[key].view(-1)):
            raise ValueError(f"Saved prediction alignment failed: {key}")
    truth = reference["target_eV"].view(-1).double().numpy()
    ref = np.abs(reference["prediction_eV"].view(-1).double().numpy() - truth)
    cand = np.abs(candidate["prediction_eV"].view(-1).double().numpy() - truth)
    if not (np.isfinite(ref).all() and np.isfinite(cand).all()):
        raise ValueError("Nonfinite saved errors")
    delta = cand - ref
    groups = []
    for index, ids in enumerate(np.array_split(np.argsort(ref, kind="stable"), 5)):
        groups.append({"reference_error_quintile": index + 1, "rows": len(ids),
            "reference_mae_eV": float(ref[ids].mean()), "candidate_mae_eV": float(cand[ids].mean()),
            "candidate_minus_reference_eV": float(delta[ids].mean())})
    return {"rows": len(delta), "reference_mae_eV": float(ref.mean()),
        "candidate_mae_eV": float(cand.mean()),
        "candidate_minus_reference_eV": float(delta.mean()),
        "paired_row_bootstrap": paired_bootstrap_mean(delta, n_bootstrap=5000, seed=20260912),
        "candidate_win_fraction": float((delta < 0).mean()),
        "absolute_error_delta_quantiles_eV": np.quantile(delta, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        "posthoc_reference_error_quintiles": groups,
        "subgroup_caveat": "Post-hoc target-derived diagnostic, not deployable routing or causal attribution"}


def accepted_native_trace(path: Path, raw: dict, checkpoint: str) -> dict:
    trace = load_canonical_trace(path)
    if len(trace["observations"]) != len(raw["epochs"]):
        raise ValueError("Native trace length differs from epoch records")
    for row, epoch in zip(trace["observations"], raw["epochs"]):
        if (row["optimizer_step"] != epoch["optimizer_steps"]
            or row["sample_presentations"] != epoch["sample_presentations"]
            or row["live_train_metric"] != epoch["train_normalized_mae"]
            or row["live_dev_metric"] != epoch["development_gap_mae_eV"]):
            raise ValueError("Native trace value differs from recorded observation")
    if trace["observations"][-1]["checkpoint_identity"] != checkpoint:
        raise ValueError("Native trace checkpoint binding differs")
    return trace


def analyze_saved_portability(reference_root: Path, candidate_root: Path, audit_root: Path,
                              mode: str) -> dict:
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    terminal = json.loads((audit_root / "terminal.json").read_text())
    result = {"format": "molgap-saved-portability-analysis-v1", "model_inference_executed": False,
        "training_executed": False, "500k_model_training_executed": False,
        "role_caveat": "Both roles are reused internal development data, not independent sealed tests"}
    result["original_100k"] = paired_saved_errors(
        load(reference_root / "best_development_payload.pt"),
        load(candidate_root / "best_development_payload.pt"))
    def joined(name):
        parts = []
        for item in terminal["unseen_500k"][name]["chunks"]:
            path = audit_root / "unseen_500k" / name / item["file"]
            if file_digest(path) != item["sha256"]:
                raise ValueError("Audit chunk hash differs")
            parts.append(load(path))
        return {key: torch.cat([part[key].view(-1) for part in parts])
                for key in ("source_idx", "target_eV", "prediction_eV")}
    result["frozen_500k"] = paired_saved_errors(joined("neural_atom_k1_v4"), joined(mode))
    raw = json.loads((candidate_root / "trace.json").read_text())["epochs"]
    base = json.loads((reference_root / "trace.json").read_text())["epochs"]
    result["matched_trajectory"] = [{"epoch_zero_based": a["epoch"],
        "optimizer_steps": a["optimizer_steps"], "candidate_train_normalized_mae": a["train_normalized_mae"],
        "reference_train_normalized_mae": b["train_normalized_mae"],
        "candidate_dev_mae_eV": a["development_gap_mae_eV"], "reference_dev_mae_eV": b["development_gap_mae_eV"],
        "candidate_minus_reference_eV": a["development_gap_mae_eV"] - b["development_gap_mae_eV"]}
        for a, b in zip(raw, base) if a["optimizer_steps"] == b["optimizer_steps"]]
    if len(result["matched_trajectory"]) != 40:
        raise ValueError("Matched late-exposure analysis requires all forty aligned observations")
    result["artifact_sha256"] = {str(path): file_digest(path) for path in (
        reference_root / "best_development_payload.pt", candidate_root / "best_development_payload.pt",
        reference_root / "trace.json", candidate_root / "trace.json", audit_root / "terminal.json")}
    return result
