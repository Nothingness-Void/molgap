"""Accept retained K1 chemistry-local training artifacts without model inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_chem_local import MODES, PARAMETERS
from molgap.research_memory.trace import load_canonical_trace
from molgap.training_reproducibility import atomic_json, sha256_file


REL = "experiments/pcqm_k1_chem_local_100k"
ROOT = REPO_ROOT / REL
RUN_ID = "kaseichou/molgap-k1-chem-local-s42:v1"
TRAJECTORIES = {
    MODES[0]: "TC-k1-atom-pair-local-100k-s42",
    MODES[1]: "TC-k1-bond-type-local-100k-s42",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accept(reference_root: Path, candidate_root: Path) -> dict:
    receipt = _json(ROOT / "submission_receipt_v1.json")
    source = _json(ROOT / "source_config.json")
    execution = _json(candidate_root / "execution_summary.json")
    if (receipt.get("submission_confirmed") is not True
        or receipt.get("physical_run_id") != RUN_ID
        or receipt.get("kernel_id") != 136210977
        or receipt.get("source_commit") != source["source_commit"]
        or not isinstance(receipt.get("source_archive_sha256"), str)
        or execution.get("run_id") != RUN_ID
        or execution.get("complete") is not True
        or execution.get("automatic_successor_submitted") is not False
        or execution.get("used_device_count") != 2
        or execution.get("allocated_device_count") != 2
        or execution.get("allocated_device_names") != ["Tesla T4", "Tesla T4"]):
        raise ValueError("Submission and completed two-device execution are not bound")
    workers = execution.get("workers", [])
    if (len(workers) != 2 or {w.get("mode") for w in workers} != set(MODES)
        or {w.get("device") for w in workers} != {0, 1}
        or any(w.get("run_id") != RUN_ID or w.get("complete") is not True
               or w.get("exit_code") != 0 or w.get("timed_out") is not False
               for w in workers)):
        raise ValueError("Independent candidate worker did not complete")

    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_chem_local_shared_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(shared)
    result = shared.accept(reference_root, candidate_root, modes=MODES,
                           expected_parameters=PARAMETERS,
                           initialization_policy="nested-function")
    reference_sha = result["reference"]["preflight"]["shared_k1_initial_state_sha256"]
    for mode in MODES:
        arm = candidate_root / mode
        row = result["candidates"][mode]
        record = row["record"]
        check = record["preflight"]["mechanism_checks"]
        trace = load_canonical_trace(arm / "canonical_trace.json")
        raw = _json(arm / "trace.json")["epochs"]
        roles = _json(arm / "observed_role_history.json")
        cost = _json(arm / "native_cost.json")
        checkpoint = torch.load(arm / "last_checkpoint.pt", map_location="cpu",
                                weights_only=False)
        if (record.get("source_commit") != source["source_commit"]
            or record["contract"]["source_archive_sha256"] != receipt["source_archive_sha256"]
            or record["contract"]["architecture_fingerprint"] != source["architecture_config_identities"][mode]
            or record["contract"]["physical_batch_per_device"] != 128
            or record["contract"]["precision"] != "fp32"
            or record["preflight"]["shared_k1_initial_state_sha256"] != reference_sha
            or record["preflight"]["preflight_memory_reserve_fraction"] < 0.15
            or check.get("target_layer") != 6
            or check.get("all_real_directed_bonds_colored") is not True
            or check.get("cross_molecule_bonds") is not False
            or check.get("distinct_active_colors") != 4
            or check.get("zero_return_projection") is not True
            or check.get("adapter_parameters") != 58_304
            or check.get("resume_two_step_bitwise_equal") is not True):
            raise ValueError(f"Chemistry-local frozen identity or mechanism changed: {mode}")
        if (trace["trajectory_id"] != TRAJECTORIES[mode]
            or trace["run_id"] != RUN_ID
            or len(trace["observations"]) != 40 or len(raw) != 40
            or trace["observations"][-1]["event"] != "terminal"
            or trace["observations"][-1]["optimizer_step"] != 31_240
            or trace["observations"][-1]["sample_presentations"] != 3_998_720
            or trace["observations"][-1]["checkpoint_identity"] != sha256_file(arm / "last_checkpoint.pt")
            or any(t["optimizer_step"] != n["optimizer_steps"]
                   or t["sample_presentations"] != n["sample_presentations"]
                   or t["live_dev_metric"] != n["development_gap_mae_eV"]
                   or t["live_train_metric"] != n["train_normalized_mae"]
                   for t, n in zip(trace["observations"], raw, strict=True))):
            raise ValueError(f"Native canonical trace incomplete: {mode}")
        if (roles.get("run_id") != RUN_ID or cost.get("run_id") != RUN_ID
            or roles.get("trajectory_id") != TRAJECTORIES[mode]
            or cost.get("trajectory_id") != TRAJECTORIES[mode]
            or any(roles.get(k) is not True for k in (
                "training_labels_read", "development_labels_read",
                "development_metric_computed", "development_selection_used"))
            or any(roles.get(k) is not False for k in (
                "official_validation_role_read", "test_dev_role_read",
                "test_challenge_role_read"))
            or cost.get("training_completed") is not True
            or not isinstance(cost.get("wall_seconds"), (int, float))
            or not math.isfinite(cost["wall_seconds"]) or cost["wall_seconds"] <= 0):
            raise ValueError(f"Native role or cost evidence incomplete: {mode}")
        if (checkpoint.get("source_commit") != source["source_commit"]
            or checkpoint.get("source_archive_sha256") != receipt["source_archive_sha256"]
            or checkpoint.get("epoch") != 39
            or any(key not in checkpoint for key in (
                "model", "optimizer", "scheduler", "rng_state", "best_epoch",
                "best_artifact_sha256"))
            or any(sha256_file(arm / name) != digest
                   for name, digest in checkpoint["best_artifact_sha256"].items())):
            raise ValueError(f"Atomic continuation checkpoint incomplete: {mode}")
        if (not all(key in checkpoint["rng_state"] for key in
                    ("python", "numpy", "torch", "cuda"))
            or not all(torch.isfinite(value).all() for value in checkpoint["model"].values())):
            raise ValueError(f"Nonfinite weights or missing RNG continuation: {mode}")
    result.update(format="molgap-k1-chem-local-acceptance-v1",
                  run_id=RUN_ID,
                  source_commit=source["source_commit"],
                  source_archive_sha256=receipt["source_archive_sha256"],
                  execution=execution,
                  model_inference_executed=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.reference_root, args.candidate_root))
