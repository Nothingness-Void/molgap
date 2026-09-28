"""Bind the exact submitted CPU sidecar to the existing server Luna monitor."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.server_control import BoundRun, LocalServerControlStore
from molgap.training_reproducibility import sha256_file


def main() -> None:
    root = REPO_ROOT / "experiments/pcqm_motif_hierarchy_100k"
    binding = json.loads((root / "monitor_binding.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "submission_receipt.json").read_text(encoding="utf-8"))
    job, = binding["jobs"]
    source = REPO_ROOT / "platforms/_records/kaggle/training/motif_hierarchy_source_e734870b"
    if binding["owner"] != "server" or binding["closed"] or job["closed"]:
        raise RuntimeError("CPU sidecar binding must be open and server-owned")
    if any(receipt[key] != job[key] for key in ("kernel", "kernel_id", "version", "run_id")):
        raise RuntimeError("CPU sidecar remote identity mismatch")
    if job["run_id"] != f"{job['kernel']}:v{job['version']}":
        raise RuntimeError("CPU sidecar run ID mismatch")
    if receipt["source_commit"] != binding["source_commit"]:
        raise RuntimeError("CPU sidecar source commit mismatch")
    if receipt["source_archive_sha256"] != binding["source_archive_sha256"]:
        raise RuntimeError("CPU sidecar archive digest mismatch")
    if (
        (source / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
        != receipt["source_commit"]
        or sha256_file(source / "source_payload.bin") != receipt["source_archive_sha256"]
        or receipt["enable_gpu"] is not False
    ):
        raise RuntimeError("CPU source/release check failed")
    bound = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5",
        chain_id="motif-hierarchy-cpu-sidecar-v1",
        run_id=job["run_id"], attempt_id="v1",
        a_thread_id=binding["controller_thread_id"],
        b_thread_id=binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=sha256_file(root / "submission_receipt.json"),
        reference_identity=None,
        budget_reserved_native={"cpu_wall_hours_upper_bound": 4.0},
        decision_ref="experiments/pcqm_motif_hierarchy_100k/protocol.md",
    ))
    print(json.dumps({"bound_run_id": bound["run_id"],
                      "monitor_generation": bound["monitor_generation"]}))


if __name__ == "__main__":
    main()
