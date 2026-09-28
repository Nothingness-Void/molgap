"""No-inference terminal acceptance for the independent MetaGIN2D screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from molgap.constants import REPO_ROOT
from molgap.k1_terminal_analysis import paired_saved_errors
from molgap.pcqm_metagin_screen import (
    ARCHITECTURE, EPOCHS, FIXED_GEOMETRY_SHA256,
    FIXED_MANIFEST_SHA256, MODEL_ID, ROW_ORDER_FINGERPRINT,
    SAMPLE_EXPOSURE, STEPS_PER_EPOCH, TRAJECTORY_ID,
)
from molgap.research_memory.trace import load_canonical_trace
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, sha256_file


ROOT = REPO_ROOT / "experiments/pcqm_metagin_2d_100k"
RUN_ID = "kaseichou/molgap-metagin-2d-s42:v1"
REFERENCE_SHA = "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sealed(record: dict, label: str) -> None:
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if record.get(role) is not False:
            raise ValueError(f"{label}: protected role {role} not sealed")


def accept(candidate: Path, reference_payload: Path) -> dict:
    source = _json(ROOT / "source_config.json")
    plan = _json(ROOT / "rml_plan/trajectory.json")
    prelaunch = _json(ROOT / "comparison_readiness_prelaunch.json")
    completion = _json(candidate / "completion_manifest.json")
    record = _json(candidate / "arm_record.json")
    preflight = _json(candidate / "preflight.json")
    certificate = _json(candidate / "runtime_certificate.json")
    roles = _json(candidate / "observed_role_history.json")
    cost = _json(candidate / "native_cost.json")
    transform = _json(candidate / "target_transform_observation.json")
    sidecar = _json(candidate / "sidecar_acceptance_binding.json")
    trace = _json(candidate / "trace.json")["epochs"]
    canonical = load_canonical_trace(candidate / "canonical_trace.json")
    checkpoint = torch.load(candidate / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    payload = torch.load(candidate / "best_development_payload.pt", map_location="cpu", weights_only=False)
    reference = torch.load(reference_payload, map_location="cpu", weights_only=False)

    if (
        prelaunch.get("prelaunch_ready") is not True
        or prelaunch.get("planned_status") != "PRELAUNCH_STRICT_PLANNED"
        or plan["trajectory_id"] != TRAJECTORY_ID
        or plan["actions"][0]["source_commit"] != source["source_commit"]
        or plan["actions"][0]["run_ids"] != [RUN_ID]
        or completion.get("complete") is not True
        or completion.get("format") != "molgap-metagin-2d-100k-completion-v1"
    ):
        raise ValueError("Prospective source/physical run/completion binding invalid")
    _sealed(completion, "completion")
    for name, digest in completion["artifact_sha256"].items():
        path = (candidate / name).resolve()
        if not path.is_relative_to(candidate.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Terminal artifact SHA mismatch: {name}")
    if set(completion["artifact_sha256"]) != {
        path.name for path in candidate.iterdir()
        if path.is_file() and path.name != "completion_manifest.json"
    }:
        raise ValueError("Completion manifest omits a terminal artifact")
    if (
        record.get("complete") is not True or record.get("model_id") != MODEL_ID
        or record.get("architecture") != ARCHITECTURE
        or record.get("architecture_config_identity") != canonical_fingerprint(ARCHITECTURE)
        or record.get("architecture_config_identity") != source["architecture_config_identity"]
        or record.get("trajectory_id") != TRAJECTORY_ID
        or record.get("run_id") != RUN_ID
        or record.get("source_commit") != source["source_commit"]
        or record.get("source_archive_sha256") != source["source_archive_sha256"]
        or record.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or record.get("fixed_geometry_sha256") != FIXED_GEOMETRY_SHA256
        or record.get("row_order_fingerprint") != ROW_ORDER_FINGERPRINT
    ):
        raise ValueError("MetaGIN scientific source/model/cache identity changed")
    _sealed(record, "arm")
    _sealed(preflight, "preflight")
    if (
        preflight.get("accepted") is not True
        or preflight.get("parameter_count") != source["model_parameters_expected"]
        or preflight.get("geometry_model_input") is not False
        or preflight.get("teacher_model_input") is not False
        or preflight.get("memory_reserve_fraction", 0) < 0.15
        or record.get("sidecar_aggregate_sha256") != sidecar.get("aggregate_sha256")
        or preflight.get("sidecar_aggregate_sha256") != sidecar.get("aggregate_sha256")
        or sidecar.get("accepted") is not True
        or sidecar.get("source_commit") != source["source_commit"]
        or sidecar.get("rows_recomputed") != 150_000
        or sidecar.get("model_inference_executed") is not False
        or sidecar.get("gap_labels_read") is not False
    ):
        raise ValueError("MetaGIN CPU sidecar or GPU preflight invalid")
    _sealed(sidecar, "sidecar")
    if (
        certificate.get("status") != "accepted"
        or certificate.get("precision") != "fp32"
        or certificate.get("tf32_enabled") is not False
        or certificate.get("deterministic_algorithms") is not True
        or certificate.get("physical_batch_per_device") != 128
        or certificate.get("calibration_checks_passed") is not True
        or certificate.get("calibration_model_id") != MODEL_ID
        or record.get("runtime_certificate_id") != canonical_fingerprint(certificate)
        or transform.get("train_target_sha256") != "df5da6a53a51d25edaacfbd72462b53a554c1ac2a667f00718a4df6308d786ce"
        or transform.get("train_source_indices_verified") is not True
        or transform.get("transform_source") != "immutable_asset_exact_values"
    ):
        raise ValueError("Runtime qualification or frozen target transform invalid")
    if (
        len(trace) != EPOCHS or len(canonical["observations"]) != EPOCHS
        or canonical["trajectory_id"] != TRAJECTORY_ID
        or canonical["run_id"] != RUN_ID
        or canonical["observations"][-1]["event"] != "terminal"
        or checkpoint.get("epoch") != EPOCHS - 1
        or checkpoint.get("source_commit") != source["source_commit"]
        or checkpoint.get("source_archive_sha256") != source["source_archive_sha256"]
        or checkpoint.get("row_order_fingerprint") != ROW_ORDER_FINGERPRINT
        or checkpoint.get("sidecar_aggregate_sha256") != sidecar["aggregate_sha256"]
        or checkpoint.get("run_id") != RUN_ID
        or checkpoint.get("trajectory_id") != TRAJECTORY_ID
        or checkpoint.get("target_transform_asset_id") != record["target_transform_asset_id"]
        or checkpoint.get("best_model_sha256") != sha256_file(candidate / "best_model.pt")
        or checkpoint.get("best_development_payload_sha256") != sha256_file(candidate / "best_development_payload.pt")
    ):
        raise ValueError("Final checkpoint/trace identity incomplete")
    for epoch, (row, observation) in enumerate(zip(trace, canonical["observations"], strict=True)):
        if (
            row["epoch"] != epoch
            or row["optimizer_steps"] != (epoch + 1) * STEPS_PER_EPOCH
            or row["sample_presentations"] != (epoch + 1) * 99_968
            or observation["optimizer_step"] != row["optimizer_steps"]
            or observation["sample_presentations"] != row["sample_presentations"]
            or observation["live_train_metric"] != row["train_normalized_mae"]
            or observation["live_dev_metric"] != row["development_gap_mae_eV"]
            or not all(math.isfinite(float(row[field])) for field in (
                "train_normalized_mae", "development_gap_mae_eV", "learning_rate", "seconds"
            ))
        ):
            raise ValueError(f"Observed optimizer/metric trace invalid at epoch {epoch}")
    if (
        canonical["observations"][-1]["checkpoint_identity"] != sha256_file(candidate / "last_checkpoint.pt")
        or trace[-1]["sample_presentations"] != SAMPLE_EXPOSURE
        or record["training"]["optimizer_steps"] != 31_240
        or record["training"]["sample_presentations"] != SAMPLE_EXPOSURE
        or record["training"]["parameter_count"] != source["model_parameters_expected"]
        or record["training"]["checkpoint_sha256"] != sha256_file(candidate / "last_checkpoint.pt")
        or not all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        or any(key not in checkpoint["rng_state"] for key in ("python", "numpy", "torch", "cuda"))
    ):
        raise ValueError("Terminal model/exposure/RNG trace invalid")
    best_epoch = min(range(EPOCHS), key=lambda index: trace[index]["development_gap_mae_eV"])
    if (
        record["training"]["best_epoch"] != best_epoch
        or checkpoint["best_epoch"] != best_epoch
        or abs(record["training"]["development_gap_mae_eV"] - trace[best_epoch]["development_gap_mae_eV"]) > 1e-8
        or not torch.equal(payload["source_idx"].view(-1), torch.arange(100_000, 150_000))
        or not bool(torch.isfinite(payload["target_eV"]).all())
        or not bool(torch.isfinite(payload["prediction_eV"]).all())
        or abs(float((payload["target_eV"] - payload["prediction_eV"]).abs().mean()) - record["training"]["development_gap_mae_eV"]) > 1e-6
    ):
        raise ValueError("Selected development payload is not the best checkpoint")
    if sha256_file(reference_payload) != REFERENCE_SHA:
        raise ValueError("Immutable K1 reference prediction payload changed")
    paired = paired_saved_errors(reference, payload)
    _sealed(roles, "observed roles")
    if (
        roles.get("trajectory_id") != TRAJECTORY_ID
        or roles.get("run_id") != RUN_ID
        or roles.get("epoch") != 39
        or any(roles.get(name) is not True for name in (
            "training_labels_read", "development_labels_read", "development_metric_computed",
            "development_selection_used",
        ))
        or cost.get("training_completed") is not True
        or cost.get("run_id") != RUN_ID
        or cost.get("model_id") != MODEL_ID
        or not math.isfinite(float(cost["allocated_device_seconds"]))
        or cost["allocated_device_seconds"] <= 0
    ):
        raise ValueError("Observed role or native cost evidence incomplete")
    gain = -paired["candidate_minus_reference_eV"]
    interval = paired["paired_row_bootstrap"]["ci95"]
    nominate = gain >= 0.003 and interval[1] < 0
    return {
        "format": "molgap-metagin-2d-100k-acceptance-v1",
        "accepted": True, "model_inference_executed": False,
        "training_executed_locally": False,
        "trajectory_id": TRAJECTORY_ID, "run_id": RUN_ID,
        "source_commit": source["source_commit"],
        "source_archive_sha256": source["source_archive_sha256"],
        "reference_payload_sha256": REFERENCE_SHA,
        "candidate_payload_sha256": sha256_file(candidate / "best_development_payload.pt"),
        "candidate_checkpoint_sha256": sha256_file(candidate / "last_checkpoint.pt"),
        "parameter_count": record["training"]["parameter_count"],
        "best_epoch": best_epoch,
        "candidate_dev_mae_eV": paired["candidate_mae_eV"],
        "reference_dev_mae_eV": paired["reference_mae_eV"],
        "gain_eV": gain, "paired": paired,
        "native_cost": cost,
        "budget_overrun": cost["wall_seconds"] > 6 * 3600,
        "nomination_gate_passed": nominate,
        "decision_scope": "seed42-screen-only-no-automatic-scale-or-protected-role",
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.candidate, args.reference_payload))
