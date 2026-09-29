"""Bind one exact version-2 physical attempt without disturbing v1 history."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.research_memory.trace import file_digest
from molgap.server_control import BoundRun, LocalServerControlStore


def main():
    root = REPO_ROOT / "experiments/pcqm_motif_hierarchy_100k/attempt_v2"
    binding = json.loads((root / "monitor_binding.json").read_text())
    receipt = json.loads((root / "submission_receipt.json").read_text())
    package = REPO_ROOT / "platforms/_records/kaggle/staging/motif_gpu_source_v2"
    release = json.loads((package / "STUDY_RELEASE.json").read_text())
    job, = binding["jobs"]
    if (binding["owner"] != "server" or binding["closed"] or job["closed"]
            or any(receipt[key] != job[key] for key in ("kernel", "kernel_id", "version", "run_id"))
            or receipt["run_id"] != release["run_id"]
            or receipt["source_commit"] != release["source_commit"]
            or receipt["source_archive_sha256"] != binding["source_archive_sha256"]
            or release["prelaunch_ready"] is not True
            or file_digest(package / "source_payload.bin") != receipt["source_archive_sha256"]):
        raise RuntimeError("Version-2 release/remote binding differs")
    bound = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5", chain_id="k1-motif-hierarchy-100k-v2",
        run_id=job["run_id"], attempt_id="v2",
        a_thread_id=binding["controller_thread_id"], b_thread_id=binding["monitor_thread_id"],
        monitor_generation=1, remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=file_digest(root / "submission_receipt.json"),
        reference_identity=file_digest(REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"),
        budget_reserved_native={"gpu_device_hours_upper_bound": 10.0},
        decision_ref="experiments/pcqm_motif_hierarchy_100k/attempt_v2/protocol.md",
    ))
    print(json.dumps({"bound_run_id": bound["run_id"],
                      "monitor_generation": bound["monitor_generation"]}))


if __name__ == "__main__":
    main()
