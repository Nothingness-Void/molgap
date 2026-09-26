"""No-model-inference acceptance of both training arms and frozen audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_linear_attention import MODES, EXPECTED_PARAMETERS as PARAMETER_COUNT
from molgap.pcqm_k1_cross_scale_diagnostic import DEVELOPMENT, MANIFESTS
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.research_memory.trace import load_canonical_trace


EXPECTED_PARAMETERS = {
    MODES[0]: PARAMETER_COUNT,
}


def _joined(output: Path, chunks: list[dict], start: int):
    if len(chunks) != 10:
        raise RuntimeError("Full 50K role requires ten atomic chunks")
    parts = []
    for index, row in enumerate(chunks):
        path = output / f"chunk_{index:02d}.pt"
        if row.get("file") != path.name or row.get("rows") != 5_000:
            raise RuntimeError("Chunk order/count changed")
        if sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"Chunk SHA mismatch: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(
            data["source_idx"].view(-1).long(),
            torch.arange(start + index * 5_000, start + (index + 1) * 5_000),
        ):
            raise RuntimeError(f"Source row order changed: {path}")
        if not all(
            torch.isfinite(data[key]).all() and data[key].numel() == 5_000
            for key in ("target_eV", "prediction_eV")
        ):
            raise RuntimeError(f"Invalid target/prediction: {path}")
        parts.append(data)
    return {
        key: torch.cat([part[key].view(-1) for part in parts])
        for key in ("source_idx", "target_eV", "prediction_eV")
    }


def accept(reference_root: Path, candidate_root: Path, source_commit: str,
           archive_sha256: str):
    shared_path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", shared_path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    training = shared.accept(
        reference_root, candidate_root, modes=MODES,
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="nested-function",
    )
    for mode in MODES:
        record = training["candidates"][mode]["record"]
        checks = record["preflight"]["mechanism_checks"]
        canonical = load_canonical_trace(candidate_root / mode / "canonical_trace.json")
        observations = canonical["observations"]
        native = json.loads((candidate_root / mode / "trace.json").read_text())["epochs"]
        if (canonical["trajectory_id"] != "TC-k1-linear-attention-100k-s42"
            or canonical["run_id"] != "nothingnessvoid/molgap-k1-linear-attention-s42:v1"
            or len(observations) != 40 or observations[-1]["event"] != "terminal"
            or observations[-1]["optimizer_step"] != 31240
            or observations[-1]["sample_presentations"] != 3998720
            or observations[-1]["checkpoint_identity"] != record["training"]["checkpoint_sha256"]):
            raise RuntimeError("Native canonical trace or checkpoint binding incomplete")
        for row, raw in zip(observations, native):
            if (row["live_dev_metric"] != raw["development_gap_mae_eV"]
                or row["live_train_metric"] != raw["train_normalized_mae"]
                or row["optimizer_step"] != raw["optimizer_steps"]
                or not row["checkpoint_identity"]):
                raise RuntimeError("Canonical observations do not match the actual training record")
        if (
            record["source_commit"] != source_commit
            or record["contract"]["source_archive_sha256"] != archive_sha256
            or checks.get("zero_start") is not True
            or checks.get("source_from_fixed_graph_only") is not True
            or checks.get("resume_two_step_bitwise_equal") is not True
            or any(checks.get(key) is not True for key in (
                "equation_verified", "permutation_equivariant", "graphs_isolated",
                "padding_invariant", "node_query_dependent", "parameter_identity",
            ))
        ):
            raise RuntimeError(f"Training identity or mechanism failed: {mode}")
    audit_root = candidate_root / "post100k_audit"
    terminal = json.loads((audit_root / "terminal.json").read_text(encoding="utf-8"))
    if (
        terminal.get("format") != "molgap-k1-post100k-portability-audit-v1"
        or terminal.get("complete") is not True
        or terminal.get("experiment_purpose") != "NO_TRAIN"
        or terminal.get("training_executed_in_audit_stage") is not False
        or terminal.get("model_inference_executed") is not True
        or terminal.get("source_commit") != source_commit
        or terminal.get("source_archive_sha256") != archive_sha256
        or terminal.get("manifest_sha256") != MANIFESTS
        or terminal.get("reference_bundle_id") != "reference-k1-v4-100k-s42-v5-recovered"
        or any(terminal.get(flag) is not False for flag in (
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"
        ))
    ):
        raise RuntimeError("NO_TRAIN audit terminal or role identity invalid")
    names = ("neural_atom_k1_v4", *MODES)
    if set(terminal["reproduction"]) != set(names) or set(terminal["unseen_500k"]) != set(names):
        raise RuntimeError("Missing or extra audit models")
    results = {}
    reference_payload = torch.load(
        reference_root / "neural_atom_k1_v4/best_development_payload.pt",
        map_location="cpu", weights_only=False,
    )
    target_500k = None
    for mode in names:
        role_path = reference_root / "neural_atom_k1_v4" if mode == names[0] else candidate_root / mode
        saved_path = role_path / "best_development_payload.pt"
        saved = reference_payload if mode == names[0] else torch.load(
            saved_path, map_location="cpu", weights_only=False
        )
        original = _joined(
            audit_root / "original_100k" / mode,
            terminal["reproduction"][mode]["chunks"],
            DEVELOPMENT["original_100k"][0],
        )
        if not torch.equal(original["target_eV"], saved["target_eV"].view(-1).float()):
            raise RuntimeError(f"Original target changed: {mode}")
        difference = float((original["prediction_eV"] - saved["prediction_eV"].view(-1).float()).abs().max())
        if difference > 0.001 or abs(difference - terminal["reproduction"][mode]["max_abs_eV"]) > 1e-7:
            raise RuntimeError(f"Checkpoint reproduction invalid: {mode}")
        unseen = _joined(
            audit_root / "unseen_500k" / mode,
            terminal["unseen_500k"][mode]["chunks"],
            DEVELOPMENT["unseen_500k"][0],
        )
        if target_500k is None:
            target_500k = unseen["target_eV"]
        elif not torch.equal(target_500k, unseen["target_eV"]):
            raise RuntimeError("500K audit target alignment failed")
        mae = float((unseen["prediction_eV"] - unseen["target_eV"]).abs().mean())
        if not math.isfinite(mae) or abs(mae - terminal["unseen_500k"][mode]["mae_eV"]) > 1e-7:
            raise RuntimeError(f"Audit metric differs from raw chunks: {mode}")
        results[mode] = {"mae_eV": mae, "reproduction_max_abs_eV": difference,
                         "checkpoint_sha256": terminal["checkpoint_sha256"][mode]}
    return {
        "format": "molgap-k1-linear-attention-acceptance-v1",
        "accepted": True,
        "training": training,
        "audit": results,
        "source_commit": source_commit,
        "source_archive_sha256": archive_sha256,
        "model_inference_executed_by_acceptance": False,
        "audit_inference_executed_in_remote_job": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "full_training_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(
        args.reference_root, args.candidate_root, args.source_commit,
        args.archive_sha256,
    ))
