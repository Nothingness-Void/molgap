"""Bind a confirmed Kaggle receipt to the existing server control store."""
from __future__ import annotations

import json
from pathlib import Path

from .constants import REPO_ROOT
from .research_memory.trace import atomic_write, json_bytes, file_digest
from .server_control import BoundRun, LocalServerControlStore, utc_timestamp
from .k1_joint_study_runtime import ATTEMPT, RUN_ID, RECIPES

REL = "experiments/pcqm_k1_joint_atom_reconstruction_100k"
A = "01a025a1-3b87-7781-8a91-f183193f7865"
B = "01a04479-ca44-7d31-95c4-6be485f256cc"


def bind(source_package, records):
    source, records = Path(source_package).resolve(), Path(records).resolve()
    root = REPO_ROOT / REL
    receipt_path = root / f"submission_receipt_v{ATTEMPT}.json"
    if receipt_path.exists():
        raise FileExistsError("Existing physical receipt cannot be overwritten")
    pushed = json.loads((records / "push_response.json").read_text())
    metadata_path = records / "remote_metadata/kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    local = json.loads((root / "kaggle/kernel-metadata.json").read_text())
    # Kaggle returns a /code/ URL path; keep the raw receipt and bind its exact slug.
    kernel = pushed["ref"].removeprefix("/code/")
    version, kernel_id = pushed["version_number"], pushed["kernel_id"]
    if (pushed.get("error") or version != ATTEMPT or kernel != local["id"]
            or (metadata["id"], metadata["id_no"]) != (kernel, kernel_id)
            or metadata["dataset_sources"] != local["dataset_sources"]
            or metadata["is_private"] is not True
            or metadata.get("machine_shape") != "NvidiaTeslaT4"):
        raise RuntimeError("Returned job/mount/accelerator identity does not match release")
    remote_entry = metadata_path.parent / metadata["code_file"]
    if remote_entry.read_text(encoding="utf-8") != (root / "kaggle/run.py").read_text(encoding="utf-8"):
        raise RuntimeError("Remote entry differs from frozen source")
    commit = (source / "SOURCE_COMMIT.txt").read_text().strip()
    archive = (source / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    actual_run = f"{kernel}:v{version}"
    control = records / "monitor/control_state.json"
    LocalServerControlStore(control).bind_run(BoundRun(
        campaign_id="k1-joint-atom-objective", chain_id="dual", run_id=actual_run,
        attempt_id=f"v{ATTEMPT}", a_thread_id=A, b_thread_id=B, monitor_generation=ATTEMPT,
        remote_platform="kaggle2", remote_job_identity={"kernel": kernel, "kernel_id": kernel_id, "version": version},
        release_identity=archive, reference_identity="reference-k1-v4-100k-s42-v5-recovered",
        budget_reserved_native={"maximum_training_and_audit_T4_hours": 12}, decision_ref=f"{REL}/protocol.md"))
    job = {"slot": "dual", "kernel": kernel, "kernel_id": kernel_id, "version": version,
        "run_id": actual_run, "embedded_trace_run_id": RUN_ID, "closed": False,
        "control_state": str(control), "output_directory": str(records),
        "candidate_root": str(records / "pcqm_k1_joint_atom_reconstruction"), "recipes": list(RECIPES)}
    receipt = {"recorded_at": utc_timestamp(), "job": job, "submission_confirmed": True,
        "raw_returned_ref": pushed["ref"],
        "source_commit": commit, "source_archive_sha256": archive,
        "source_dataset": "kaseichou/molgap-k1-joint-atom-source", "source_dataset_private": True,
        "source_dataset_status_at_release": "ready", "returned_machine_shape": metadata["machine_shape"],
        "actual_hardware": "pending_startup_log", "remote_metadata_sha256": file_digest(metadata_path),
        "remote_entry_sha256": file_digest(remote_entry), "scientific_entry_text_matches": True,
        "runtime_trace_identity_rewritten": False, "protected_roles_accessed": False}
    atomic_write(receipt_path, json_bytes(receipt))
    binding = {"format": "molgap-server-multi-job-monitor-v1", "owner": "server", "closed": False,
        "working_directory": str(REPO_ROOT), "python": str(REPO_ROOT / ".venv/Scripts/python.exe"),
        "credential_file": "C:/Users/Adminn/Desktop/kaggle.json", "credential_owner": "kaseichou",
        "controller_thread_id": A, "monitor_thread_id": B, "source_commit": commit,
        "source_archive_sha256": archive, "jobs": [job], "healthy_action": "SILENT",
        "terminal_action": "one idempotent handoff to existing A; no model/thinking override",
        "automatic_training_retry": False, "automatic_training_successor_authorized": False,
        "acceptance_script": str(root / "accept.py")}
    atomic_write(root / "monitor_binding.json", json_bytes(binding))
    return receipt
