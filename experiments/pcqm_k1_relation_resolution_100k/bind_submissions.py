"""Record the two confirmed v1 pushes and bind the existing local Luna monitor."""
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.k1_relation_study_runtime import RUNS, SLOTS
from molgap.research_memory.trace import atomic_write, json_bytes, file_digest
from molgap.server_control import BoundRun, LocalServerControlStore, utc_timestamp

REL = "experiments/pcqm_k1_relation_resolution_100k"
A = "01a025a1-3b87-7781-8a91-f183193f7865"
B = "01a04479-ca44-7d31-95c4-6be485f256cc"
REMOTE = {"dual": ("kaseichou/molgap-k1-receiver-and-triplet-s42", 136015255),
          "rrwp": ("kaseichou/molgap-k1-rrwp-pair-s42", 136015256)}


def main():
    root = REPO_ROOT / REL
    receipt_path = root / "submission_receipt_v1.json"
    if receipt_path.exists():
        raise FileExistsError("Do not reset a monitor or overwrite a submission receipt")
    source = REPO_ROOT / "platforms/_staging/kaggle/k1_relation_resolution_source_v1"
    commit = (source / "SOURCE_COMMIT.txt").read_text().strip()
    archive = (source / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    jobs, receipts = [], []
    for slot, (kernel, kernel_id) in REMOTE.items():
        record_root = REPO_ROOT / f"platforms/_records/kaggle/training/k1_relation_resolution_{slot}_v1"
        metadata_path = record_root / "remote_metadata/kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (metadata["id"], metadata["id_no"], metadata["is_private"]) != (kernel, kernel_id, True):
            raise ValueError("Downloaded scheduler identity is inconsistent with the push receipt")
        if metadata["dataset_sources"] != ["kaseichou/molgap-k1-relation-resolution-source", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"]:
            raise ValueError("Remote mounts differ from released datasets")
        entry = root / f"kaggle_{slot}/run.py"
        remote_entry = metadata_path.parent / metadata["code_file"]
        if remote_entry.read_text(encoding="utf-8") != entry.read_text(encoding="utf-8"):
            raise ValueError("Pulled remote entry differs from the released entry")
        actual_run = f"{kernel}:v1"
        state_path = record_root / "monitor/control_state.json"
        LocalServerControlStore(state_path).bind_run(BoundRun(
            campaign_id="k1-relation-resolution-20260927", chain_id=slot,
            run_id=actual_run, attempt_id="v1", a_thread_id=A, b_thread_id=B,
            monitor_generation=1, remote_platform="kaggle2",
            remote_job_identity={"kernel": kernel, "kernel_id": kernel_id, "version": 1},
            release_identity=archive, reference_identity="reference-k1-v4-100k-s42-v5-recovered",
            budget_reserved_native={"max_active_training_device_hours": len(SLOTS[slot]) * 6},
            decision_ref=f"{REL}/protocol.md"))
        receipts.append({"slot": slot, "kernel": kernel, "kernel_id": kernel_id, "version": 1,
            "submission_confirmed": True, "actual_physical_run_id": actual_run,
            "embedded_trace_run_id": RUNS[slot],
            "run_alias_reason": "Kaggle generated its slug from the title on first creation" if slot == "dual" else None,
            "runtime_trace_identity_rewritten": False,
            "source_commit": commit, "source_archive_sha256": archive,
            "remote_metadata_sha256": file_digest(metadata_path),
            "remote_entry_sha256": file_digest(remote_entry),
            "scientific_entry_text_matches": True,
            "returned_machine_shape": metadata.get("machine_shape"),
            "actual_hardware": "pending_startup_log",
            "outputs_directory": record_root.relative_to(REPO_ROOT).as_posix()})
        jobs.append({"slot": slot, "kernel": kernel, "version": 1, "kernel_id": kernel_id,
            "run_id": actual_run, "embedded_trace_run_id": RUNS[slot], "closed": False,
            "control_state": str(state_path), "output_directory": str(record_root),
            "candidate_root": str(record_root / "pcqm_k1_relation_resolution"),
            "modes": list(SLOTS[slot])})
    atomic_write(receipt_path, json_bytes({"recorded_at": utc_timestamp(), "jobs": receipts,
        "source_dataset": "kaseichou/molgap-k1-relation-resolution-source",
        "source_dataset_status_at_release": "ready", "source_dataset_private": True,
        "upload_note": "First CLI upload failed locally before dataset creation due to slash handling; normalized Windows path succeeded. No training retry.",
        "protected_roles_accessed": False}))
    atomic_write(root / "monitor_binding.json", json_bytes({
        "format": "molgap-server-multi-job-monitor-v1", "owner": "server", "closed": False,
        "working_directory": str(REPO_ROOT), "python": str(REPO_ROOT / ".venv/Scripts/python.exe"),
        "credential_file": "C:/Users/Adminn/Desktop/kaggle.json", "credential_owner": "kaseichou",
        "controller_thread_id": A, "monitor_thread_id": B,
        "source_commit": commit, "source_archive_sha256": archive,
        "reference_root": str(REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference"),
        "acceptance_script": str(root / "accept.py"), "jobs": jobs,
        "first_terminal_does_not_close_other_job": True,
        "terminal_action": "one idempotent handoff per job to existing A; no model/thinking override",
        "healthy_action": "SILENT", "automatic_training_retry": False,
        "no_train_audit_requires_controller_training_acceptance": True}))
    print(json.dumps({"receipt": str(receipt_path), "bound_jobs": [r["kernel"] for r in receipts]}))


if __name__ == "__main__":
    main()
