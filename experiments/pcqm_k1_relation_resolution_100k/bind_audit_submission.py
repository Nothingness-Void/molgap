"""Bind the confirmed NO_TRAIN audit without replacing training provenance."""
import json

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.server_control import BoundRun, LocalServerControlStore, utc_timestamp

REL = "experiments/pcqm_k1_relation_resolution_100k"


def main():
    root = REPO_ROOT / REL
    receipt_path = root / "audit_submission_receipt_v1.json"
    if receipt_path.exists():
        raise FileExistsError("Never reset an existing audit binding or receipt")
    training_binding = json.loads((root / "monitor_binding.json").read_text())
    release_path = root / "audit_release.json"
    release = json.loads(release_path.read_text())
    records = REPO_ROOT / "platforms/_records/kaggle/training/k1_relation_resolution_audit_v1"
    metadata_path = records / "remote_metadata/kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = json.loads((root / "kaggle_audit/kernel-metadata.json").read_text())
    if (metadata["id"], metadata["id_no"], metadata["is_private"]) != (
            "kaseichou/molgap-k1-relation-audit-s42", 136030465, True):
        raise ValueError("Scheduler identity differs from the confirmed v1 push")
    if sorted(metadata["dataset_sources"]) != sorted(expected["dataset_sources"]):
        raise ValueError("Audit mounts changed")
    remote_entry = metadata_path.parent / metadata["code_file"]
    if remote_entry.read_text() != (root / "kaggle_audit/run.py").read_text():
        raise ValueError("Remote audit entry changed")
    if release["training_authorized"] is not False:
        raise ValueError("Only the NO_TRAIN audit was released")
    state_path = records / "monitor/control_state.json"
    job = {"slot": "audit", "kernel": metadata["id"], "kernel_id": metadata["id_no"],
        "version": 1, "run_id": release["run_id"], "closed": False,
        "control_state": str(state_path), "output_directory": str(records),
        "audit_root": str(records / "pcqm_k1_relation_audit")}
    LocalServerControlStore(state_path).bind_run(BoundRun(
        campaign_id="k1-relation-resolution-20260927", chain_id="audit",
        run_id=release["run_id"], attempt_id="v1",
        a_thread_id=training_binding["controller_thread_id"],
        b_thread_id=training_binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=release["input_archive_sha256"],
        reference_identity="reference-k1-v4-100k-s42-v5-recovered",
        budget_reserved_native={"max_allocated_device_seconds": 5400},
        decision_ref=f"{REL}/audit_release.json"))
    atomic_write(receipt_path, json_bytes({
        "recorded_at": utc_timestamp(), "submission_confirmed": True, **job,
        "training_authorized": False, "reference_inference_reused": True,
        "release_sha256": file_digest(release_path),
        "remote_metadata_sha256": file_digest(metadata_path),
        "remote_entry_sha256": file_digest(remote_entry), "entry_text_matches": True,
        "source_commit": release["source_commit"],
        "source_archive_sha256": release["source_archive_sha256"],
        "audit_helper_commit": release["audit_helper_commit"],
        "input_archive_sha256": release["input_archive_sha256"],
        "input_dataset_ready_verified": True,
        "returned_machine_shape": metadata.get("machine_shape"),
        "status_observed_after_push": "RUNNING", "actual_device_pending": True}))
    binding = {key: training_binding[key] for key in (
        "format", "owner", "working_directory", "python", "credential_file",
        "credential_owner", "controller_thread_id", "monitor_thread_id",
        "source_commit", "source_archive_sha256", "healthy_action", "terminal_action")}
    binding.update(closed=False, jobs=[job], experiment_purpose="NO_TRAIN",
        automatic_training_retry=False, acceptance_script=str(root / "accept_audit.py"))
    atomic_write(root / "audit_monitor_binding.json", json_bytes(binding))
    print(json.dumps({"receipt": str(receipt_path), "job": job}))


if __name__ == "__main__":
    main()
