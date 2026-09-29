"""Freeze the T4-only retry using the existing reference-bound release gate."""
from molgap.k1_motif_study_runtime import RUN_ID, TRAJECTORY_ID

from .freeze_gpu_release_v2 import freeze_retry


if __name__ == "__main__":
    freeze_retry(
        version=3, run_id=RUN_ID, trajectory_id=TRAJECTORY_ID,
        infrastructure_change="request_T4_and_isolate_one_of_up_to_two_visible_T4s",
        device_hours_ceiling=20,
    )
