"""Streaming, no-checkpoint-load acceptance of GPTrans V5 audit segments."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH, EPOCHS, EXPECTED_PARAMETERS,
    MANIFEST_SHA256, PHYSICAL_BATCH, RUN_FORMAT,
)
from .research_memory.trace import load_canonical_trace
from .training_reproducibility import atomic_json, sha256_file


SOURCE_SHA256 = "af94a63c3c4626ceec8af4106aad0ea97d398e40aefb04c51b15356b994137a3"
SOURCE_COMMIT = "21f70ab93c3b0ddc4745316daf8459df40efdb48"
TRAJECTORY = "TC-gptrans-v5-audit-reference-100k"
RUN = "gptrans-t-v5-audit-reference-s42"
RESUME_FILES = (
    "last_checkpoint.pt", "trace.json", "canonical_trace.json",
    "best_model.pt", "development_predictions.pt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid JSON object: {path.name}")
    return value


def accept_segment(root: Path, expected_epochs: int, acceptance_path: Path) -> dict:
    """Accept a segment's observed state without loading model tensors into RAM."""
    root = root.resolve()
    require(expected_epochs in (10, 20, 30, 40, 50, 60), "Segment epoch boundary")
    training = root / "training"
    preflight = load(root / "preflight" / "preflight.json")
    cost = load(root / "native_cost.json")
    manifest = load(training / ("completion_manifest.json" if expected_epochs == EPOCHS else "partial_manifest.json"))
    trace_rows = load(training / "trace.json")["rows"]
    trace = load_canonical_trace(training / "canonical_trace.json")
    require(preflight.get("accepted") is True and preflight.get("variant") == "reference", "Preflight acceptance")
    require(preflight.get("parameters") == EXPECTED_PARAMETERS, "Model parameter count")
    require(preflight.get("manifest_sha256") == MANIFEST_SHA256, "Preflight fixed data")
    require(preflight.get("source_archive_sha256") == SOURCE_SHA256, "Preflight source")
    require(preflight.get("source_commit") == SOURCE_COMMIT, "Preflight source commit")
    certificate = preflight["runtime_certificate"]
    require(certificate.get("status") == "accepted", "Runtime certificate status")
    require(certificate.get("physical_batch_per_device") == PHYSICAL_BATCH, "Runtime batch")
    require(certificate.get("precision") == "fp32" and certificate.get("tf32_enabled") is False, "Runtime precision")
    require("T4" in certificate.get("accelerator", ""), "Runtime T4 identity")
    require(manifest.get("format") == RUN_FORMAT and manifest.get("v5_audit") is True, "Audit manifest")
    require(manifest.get("complete") is (expected_epochs == EPOCHS), "Completion state")
    require(manifest.get("manifest_sha256") == MANIFEST_SHA256, "Manifest fixed data")
    require(manifest.get("source_archive_sha256") == SOURCE_SHA256, "Manifest source")
    require(manifest.get("runtime_certificate_id") == preflight["runtime_certificate_id"], "Certificate identity")
    if expected_epochs < EPOCHS:
        require(manifest.get("completed_epochs") == manifest.get("next_epoch") == expected_epochs, "Partial epoch cursor")
        require(set(manifest.get("resume_file_sha256", {})) == set(RESUME_FILES), "Resume file inventory")
        for name in RESUME_FILES:
            require(sha256_file(training / name) == manifest["resume_file_sha256"][name], f"Resume hash {name}")
    else:
        require(manifest.get("epochs") == EPOCHS, "Final epoch count")
        require(manifest.get("optimizer_steps") == BATCHES_PER_EPOCH * EPOCHS, "Final optimizer steps")
        require(manifest.get("sample_presentations") == BATCHES_PER_EPOCH * PHYSICAL_BATCH * EPOCHS, "Final sample exposure")
        for name, key in (("best_model.pt", "best_model_sha256"), ("development_predictions.pt", "development_predictions_sha256")):
            require(sha256_file(training / name) == manifest[key], f"Final artifact hash {name}")
        for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
            require(manifest.get(role) is False, f"Protected role {role}")
    require(sha256_file(training / "last_checkpoint.pt") == manifest["checkpoint_sha256"], "Last checkpoint hash")
    require(sha256_file(training / "canonical_trace.json") == manifest["canonical_trace_sha256"], "Canonical trace hash")
    chunk = training / f"checkpoint_epoch_{expected_epochs - 1:02d}.pt"
    require(chunk.is_file() and sha256_file(chunk) == manifest["checkpoint_sha256"], "Bounded checkpoint chunk")
    require(trace.get("trajectory_id") == TRAJECTORY and trace.get("run_id") == RUN, "Canonical trace identity")
    observations = trace["observations"]
    require(len(observations) == len(trace_rows) == expected_epochs, "Trace row count")
    for i, (row, observation) in enumerate(zip(trace_rows, observations)):
        expected_steps = (i + 1) * BATCHES_PER_EPOCH
        expected_presentations = expected_steps * PHYSICAL_BATCH
        require(row.get("epoch") == i and observation["epoch_or_pass"] == i + 1, "Epoch order")
        require(row.get("cumulative_optimizer_steps") == observation["optimizer_step"] == expected_steps, "Optimizer exposure")
        require(row.get("cumulative_sample_presentations") == observation["sample_presentations"] == expected_presentations, "Sample exposure")
        for legacy, canonical in (("train_mae_eV", "live_train_metric"), ("live_development_mae_eV", "live_dev_metric"), ("development_mae_eV", "ema_dev_metric")):
            value = row.get(legacy)
            require(isinstance(value, (int, float)) and math.isfinite(value), f"Finite {legacy}")
            require(value == observation[canonical], f"Trace metric {canonical}")
        require(row.get("learning_rate") == observation["learning_rate"], "Learning rate observation")
        require(observation["event"] == "checkpoint" and observation["checkpoint_identity"], "Observed checkpoint")
        require(observation["cumulative_wall_time_seconds"] == row["cumulative_wall_time_seconds"], "Wall time observation")
    require(observations[-1]["checkpoint_identity"] == f"sha256:{manifest['checkpoint_sha256']}", "Last trace/checkpoint binding")
    require(cost.get("source_archive_sha256") == SOURCE_SHA256 and cost.get("source_commit") == SOURCE_COMMIT, "Cost source")
    require(cost.get("completed_epochs") == expected_epochs, "Cost epoch cursor")
    allocation = cost.get("allocated_gpu_inventory")
    require(isinstance(allocation, list) and len(allocation) == cost.get("allocated_gpu_count"), "Allocated GPU inventory")
    require(all("T4" in item for item in allocation) and cost.get("used_gpu_count") == 1, "GPU allocation accounting")
    require(cost.get("wall_seconds", 0) > 0, "Measured wall cost")
    result = {
        "format": "molgap-gptrans-v5-audit-segment-acceptance-v1",
        "accepted": True, "completed_epochs": expected_epochs,
        "next_segment_authorized": expected_epochs < EPOCHS,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "canonical_trace_sha256": manifest["canonical_trace_sha256"],
        "runtime_certificate_id": preflight["runtime_certificate_id"],
        "source_archive_sha256": SOURCE_SHA256,
        "allocated_device_hours": cost["wall_seconds"] * len(allocation) / 3600.0,
        "best_ema_development_mae_eV": min(row["development_mae_eV"] for row in trace_rows),
        "final_prediction_payload_content_checked": False,
    }
    atomic_json(acceptance_path, result)
    return result
