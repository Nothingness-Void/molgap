"""Bind the exact submitted MetaGIN2D screen to the existing server monitor."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.server_control import BoundRun, LocalServerControlStore
from molgap.training_reproducibility import sha256_file


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = REPO_ROOT / "experiments/pcqm_metagin_2d_100k"
    frozen = root / "attempt_v2"
    binding = _read(root / "monitor_binding.json")
    receipt = _read(frozen / "submission_receipt.json")
    source_config = _read(frozen / "source_config.json")
    prelaunch = _read(frozen / "comparison_readiness_prelaunch.json")
    trajectory = _read(frozen / "rml_plan/trajectory.json")
    job, = binding["jobs"]
    source = REPO_ROOT / "platforms/_records/kaggle/metagin_2d_100k_source_v4"
    if binding["owner"] != "server" or binding["closed"] or job["closed"]:
        raise RuntimeError("Screen monitor must remain open and server-owned")
    if (
        receipt["physical_run_id"] != job["run_id"]
        or receipt["kernel"] != job["kernel"]
        or receipt["kernel_id"] != job["kernel_id"]
        or receipt["version"] != job["version"]
        or job["run_id"] != f"{job['kernel']}:v{job['version']}"
        or receipt["source_commit"] != binding["source_commit"]
        or receipt["source_archive_sha256"] != binding["source_archive_sha256"]
        or receipt["source_commit"] != source_config["source_commit"]
        or receipt["source_archive_sha256"] != source_config["source_archive_sha256"]
        or source_config["gpu_run_id"] != job["run_id"]
        or receipt["trajectory_id"] != trajectory["trajectory_id"]
        or trajectory["actions"][0]["run_ids"] != [job["run_id"]]
        or prelaunch.get("prelaunch_ready") is not True
        or prelaunch.get("planned_status") != "PRELAUNCH_STRICT_PLANNED"
    ):
        raise RuntimeError("Submitted screen differs from prospective release")
    if (
        (source / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
        != receipt["source_commit"]
        or (source / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
        != receipt["source_archive_sha256"]
        or sha256_file(source / "source_payload.bin") != receipt["source_archive_sha256"]
    ):
        raise RuntimeError("Uploaded executable source does not match release")
    state = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5",
        chain_id="metagin-2d-independent-screen-v2",
        run_id=job["run_id"], attempt_id="v2",
        a_thread_id=binding["controller_thread_id"],
        b_thread_id=binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=sha256_file(frozen / "comparison_readiness_prelaunch.json"),
        reference_identity=trajectory["reference_bundle_id"],
        budget_reserved_native={"gpu_device_hours_upper_bound": 12.0},
        decision_ref="experiments/pcqm_metagin_2d_100k/results/profile_v2_decision.md",
    ))
    print(json.dumps({
        "bound_run_id": state["run_id"],
        "monitor_generation": state["monitor_generation"],
    }))


if __name__ == "__main__":
    main()
