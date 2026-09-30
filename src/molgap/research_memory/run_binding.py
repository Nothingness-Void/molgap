"""Bind a scheduler-assigned diagnostic ID without rewriting its frozen plan.

This is deliberately not a training-run override or a causal/replay qualification.
Local receipts are provenance evidence, not a cryptographic remote attestation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import verify_bound_artifact


def verify_diagnostic_run_binding(
    root: Path, frozen: dict[str, Any], frozen_bytes: bytes,
    package: dict[str, Any], *, has_trace: bool,
) -> dict[str, Any]:
    """Fail closed outside a completed, source-bound NO_TRAIN diagnostic."""
    action = next((a for a in frozen["actions"]
                   if a["action_id"] == package["action_id"]), None)
    # Runtime import avoids coupling the low-level module to orchestration imports.
    from .terminal_wiring import inspect_trace_retention_evidence
    declared_trace, _ = inspect_trace_retention_evidence(package["evidence"])
    if (frozen["owner"] != "server" or frozen["record_mode"] != "prospective"
            or action is None or action["run_ids"]
            or "NO_TRAIN" not in action["type"]
            or package["decision"]["outcome"] != "NO_TRAIN"
            or package["evidence"]["outcome"]["comparison_status"] != "context_only"
            or package["evidence"]["outcome"]["execution_status"] != "complete"
            or has_trace or declared_trace or package.get("trace_manifest") is not None
            or package.get("action_replay") is not None
            or package.get("comparison_readiness_ref") is not None):
        raise ValueError("postlaunch binding is restricted to completed NO_TRAIN context diagnostics")

    def bound(pointer: str, digest: str | None = None) -> dict[str, Any]:
        recorded = package["artifact_hashes"].get(pointer)
        if recorded is None or (digest is not None and digest != recorded):
            raise ValueError("postlaunch receipt is not hash bound")
        verify_bound_artifact(root, pointer, recorded)
        return load_json_object(root / pointer)

    binding = bound(package["postlaunch_run_binding_ref"])
    expected = {
        "format": "molgap-no-train-postlaunch-binding-v1",
        "trajectory_id": frozen["trajectory_id"],
        "action_id": action["action_id"], "run_id": package["run_id"],
        "original_trajectory_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    }
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("postlaunch binding differs from frozen plan/run")
    submission = bound(binding["submission_ref"], binding["submission_sha256"])
    scheduler = bound(binding["scheduler_ref"], binding["scheduler_sha256"])
    started = bound(binding["started_ref"], binding["started_sha256"])
    source = frozen["state_at_start"]["source_commit"]
    if (action["source_commit"] != source
            or submission.get("format") != "molgap-representation-submission-v1"
            or submission.get("owner") != "server"
            or submission.get("purpose") != "NO_TRAIN"
            or submission.get("source_commit") != source
            or started.get("source_commit") != source
            or any(r.get("job_id") != package["run_id"]
                   for r in (submission, scheduler, started))
            or scheduler.get("state") != "COMPLETED"
            or scheduler.get("exit_code") != "0:0"
            or started.get("training_executed") is not False
            or started.get("model_inference_executed") is not True):
        raise ValueError("postlaunch receipts do not identify the completed frozen diagnostic")
    archive = submission.get("source_archive_sha256")
    if (not isinstance(archive, str) or len(archive) != 64
            or any(c not in "0123456789abcdef" for c in archive)
            or started.get("source_archive_sha256") != archive):
        raise ValueError("postlaunch source archive identity mismatch")
    for field in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if started.get(field) is not False:
            raise ValueError("postlaunch diagnostic protected-role truth is not explicit")
    return binding
