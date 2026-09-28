"""Bind one exact already-submitted profile to the existing server A/B loop."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.server_control import BoundRun, LocalServerControlStore
from molgap.training_reproducibility import sha256_file


def main() -> None:
    root = REPO_ROOT / "experiments/pcqm_metagin_2d_100k"
    binding = json.loads((root / "monitor_binding.json").read_text(encoding="utf-8"))
    job, = binding["jobs"]
    source = REPO_ROOT / "platforms/_records/kaggle/metagin_2d_100k_source_v3"
    if binding["owner"] != "server" or binding["closed"] or job["closed"]:
        raise RuntimeError("Profile monitor must remain open and server-owned")
    if job["run_id"] != f"{job['kernel']}:v{job['version']}":
        raise RuntimeError("Profile run/version binding differs")
    if (
        (source / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
        != binding["source_commit"]
        or (source / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
        != binding["source_archive_sha256"]
        or sha256_file(source / "source_payload.bin") != binding["source_archive_sha256"]
    ):
        raise RuntimeError("Profile executable source is not the uploaded source")
    state = LocalServerControlStore(Path(job["control_state"])).bind_run(BoundRun(
        campaign_id="molgap-pcqm-gap-v5",
        chain_id="metagin-2d-runtime-profile-v2",
        run_id=job["run_id"], attempt_id="v2",
        a_thread_id=binding["controller_thread_id"],
        b_thread_id=binding["monitor_thread_id"], monitor_generation=1,
        remote_platform="kaggle2",
        remote_job_identity={key: job[key] for key in ("kernel", "kernel_id", "version")},
        release_identity=binding["source_archive_sha256"],
        reference_identity=None,
        budget_reserved_native={"gpu_device_hours_upper_bound": 1.0},
        decision_ref="experiments/pcqm_metagin_2d_100k/runtime_profile_protocol.md",
    ))
    print(json.dumps({
        "bound_run_id": state["run_id"],
        "monitor_generation": state["monitor_generation"],
    }))


if __name__ == "__main__":
    main()
