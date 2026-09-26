"""Observed-only RML telemetry for the new single-device K1 screen."""
from .research_memory.trace import RMLTraceRecorder
from .training_reproducibility import atomic_json, sha256_file


def recorder(output, trajectory_id, run_id):
    def metric(unit, role):
        return {"metric": "mean_absolute_error", "unit": unit, "target": "gap",
                "role_identity": role, "weights": "live", "direction": "minimize"}
    return RMLTraceRecorder(
        output / "canonical_trace.json", trajectory_id=trajectory_id, run_id=run_id,
        metric_semantics={
            "live_train_metric": metric("normalized_target_units", "official_train_prefix_0_100000"),
            "live_dev_metric": metric("eV", "internal_development_100000_150000"),
            "ema_dev_metric": None,
        }, device_time_semantics="sum_over_devices",
    )


def record_epoch(recorder, output, row, *, observed_steps, observed_samples, elapsed):
    previous = recorder.record["observations"]
    cumulative = elapsed + (previous[-1]["cumulative_device_time_seconds"] if previous else 0.0)
    recorder.append_observation(
        event="terminal" if row["epoch"] == 39 else "checkpoint",
        optimizer_step=observed_steps, sample_presentations=observed_samples,
        epoch_or_pass=row["epoch"], learning_rate=row["learning_rate"],
        live_train_metric=row["train_normalized_mae"],
        live_dev_metric=row["development_gap_mae_eV"],
        checkpoint_identity=sha256_file(output / "last_checkpoint.pt"),
        wall_time_seconds=elapsed, cumulative_wall_time_seconds=cumulative,
        device_time_seconds=elapsed, cumulative_device_time_seconds=cumulative,
    )
    atomic_json(output / "observed_role_history.json", {
        "format": "molgap-observed-k1-screen-roles-v1", "epoch": row["epoch"],
        "trajectory_id": recorder.record["trajectory_id"], "run_id": recorder.record["run_id"],
        "training_membership": "official_train_prefix_0_100000",
        "training_labels_read": True,
        "development_prediction_input": "internal_development_100000_150000",
        "development_labels_read": True, "development_metric_computed": True,
        "development_selection_used": True, "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })
