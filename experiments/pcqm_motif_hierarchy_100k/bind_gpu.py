"""Bind the exact submitted GPU attempt to the existing server Luna B."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import file_digest
from molgap.server_control import BoundRun, LocalServerControlStore


def main():
    root = REPO_ROOT / "experiments/pcqm_motif_hierarchy_100k"
    binding = json.loads((root / "gpu_monitor_binding.json").read_text())
    receipt = json.loads((root / "gpu_submission_receipt.json").read_text())
    release = json.loads((REPO_ROOT / "platforms/_records/kaggle/staging/motif_gpu_source_v1/STUDY_RELEASE.json").read_text())
    job, = binding["jobs"]
    if binding["owner"] != "server" or binding["closed"] or job["closed"]:
        raise RuntimeError("GPU binding must be open and server-owned")
    if any(receipt[key] != job[key] for key in ("kernel", "kernel_id", "version", "run_id")):
        raise RuntimeError("GPU remote identity mismatch")
    if (receipt["source_commit"] != release["source_commit"]
            or receipt["source_archive_sha256"] != binding["source_archive_sha256"]
            or receipt["source_commit"] != binding["source_commit"]
            or receipt["run_id"] != release["run_id"]
            or release["prelaunch_ready"] is not True):
        raise RuntimeError("GPU source/prelaunch release mismatch")
    bound = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5",
        chain_id="k1-motif-hierarchy-100k-v1",
        run_id=job["run_id"], attempt_id="v1",
        a_thread_id=binding["controller_thread_id"],
        b_thread_id=binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=file_digest(root / "gpu_submission_receipt.json"),
        reference_identity=file_digest(REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"),
        budget_reserved_native={"gpu_device_hours_upper_bound": 10.0},
        decision_ref="experiments/pcqm_motif_hierarchy_100k/gpu_protocol.md",
    ))
    print(json.dumps({"bound_run_id": bound["run_id"],
                      "monitor_generation": bound["monitor_generation"]}))


if __name__ == "__main__":
    main()
