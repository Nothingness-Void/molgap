"""Bind the already-submitted exact Kaggle version to the existing A/B loop."""
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import file_digest
from molgap.server_control import BoundRun, LocalServerControlStore


def main():
    root = REPO_ROOT / "experiments/pcqm_k1_chem_local_100k"
    binding = json.loads((root / "monitor_binding.json").read_text())
    receipt = json.loads((root / "submission_receipt_v1.json").read_text())
    if binding["owner"] != "server" or binding["closed"] or len(binding["jobs"]) != 1:
        raise RuntimeError("Monitor owner/closure changed")
    job = binding["jobs"][0]
    if (receipt["physical_run_id"] != job["run_id"]
            or receipt["kernel_id"] != job["kernel_id"]
            or receipt["version"] != job["version"]
            or receipt["source_commit"] != binding["source_commit"]
            or receipt["source_archive_sha256"] != binding["source_archive_sha256"]):
        raise RuntimeError("Remote receipt differs from frozen monitor binding")
    source_release = REPO_ROOT / "platforms/_records/kaggle/source/k1_chem_local_v1/STUDY_RELEASE.json"
    release = json.loads(source_release.read_text())
    if release["run_id"] != job["run_id"] or release["source_commit"] != binding["source_commit"]:
        raise RuntimeError("Prospective source release differs")
    state = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5", chain_id="k1-chem-local-seed42",
        run_id=job["run_id"], attempt_id="v1", a_thread_id=binding["controller_thread_id"],
        b_thread_id=binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2", remote_job_identity={"kernel": job["kernel"],
            "kernel_id": job["kernel_id"], "version": job["version"]},
        release_identity=file_digest(source_release),
        reference_identity="reference-k1-v4-100k-s42-v5-recovered",
        budget_reserved_native={"t4_device_hours_upper_bound": 12.0},
        decision_ref="experiments/pcqm_k1_chem_local_100k/decision.md"))
    print(json.dumps({"bound_run_id": state["run_id"], "monitor_generation": state["monitor_generation"]}))


if __name__ == "__main__":
    main()
