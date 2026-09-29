"""No-GPU checks for the additive GPTrans reference audit path."""

from pathlib import Path

import torch

from molgap import pcqm_gptrans_v4 as baseline
from molgap.gptrans_v5_audit_acceptance import (
    RESUME_FILES, RUN, SOURCE_COMMIT, SOURCE_SHA256, TRAJECTORY, accept_segment,
)
from molgap.research_memory.trace import load_canonical_trace
from molgap.training_reproducibility import atomic_json, sha256_file


class _Batch:
    num_graphs = 2

    def __init__(self):
        self.x = torch.ones((2, 1))
        self.y = torch.zeros(2)
        self.source_idx = torch.tensor([100_000, 100_001])

    def to(self, *args, **kwargs):
        return self


def test_live_and_ema_development_observations_keep_model_state(monkeypatch):
    monkeypatch.setattr(baseline, "DEVELOPMENT_ROWS", 2)
    monkeypatch.setattr(baseline, "_development_loader", lambda graphs: [_Batch()])
    monkeypatch.setattr(baseline, "_forward", lambda model, batch: model(batch.x).view(-1))
    live = torch.nn.Linear(1, 1, bias=False)
    ema = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        live.weight.fill_(1.0)
        ema.weight.fill_(2.0)
    original = live.weight.detach().clone()
    default = baseline._evaluate(live, ema, None, torch.tensor(0.0), torch.tensor(1.0), device="cpu")
    observed_live = baseline._evaluate(live, ema, None, torch.tensor(0.0), torch.tensor(1.0), weights="live", device="cpu")
    assert default["mae_eV"] == 2.0
    assert observed_live["mae_eV"] == 1.0
    assert torch.equal(live.weight, original)
    assert torch.equal(default["source_idx"], observed_live["source_idx"])


def test_v5_trace_checkpoint_observation_is_resumable(tmp_path: Path):
    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    trace_path = tmp_path / "canonical_trace.json"
    row = {
        "epoch": 0,
        "cumulative_optimizer_steps": 781,
        "cumulative_sample_presentations": 99_968,
        "learning_rate": 0.00025,
        "train_mae_eV": 0.4,
        "live_development_mae_eV": 0.5,
        "development_mae_eV": 0.45,
        "elapsed_seconds": 12.0,
        "cumulative_wall_time_seconds": 12.0,
    }
    recorder = baseline._v5_audit_recorder(trace_path)
    baseline._v5_append_checkpoint_observation(recorder, row, checkpoint)
    resumed = baseline._v5_audit_recorder(trace_path)
    baseline._v5_append_checkpoint_observation(resumed, row, checkpoint)
    rows = load_canonical_trace(trace_path)["observations"]
    assert len(rows) == 1
    assert rows[0]["live_dev_metric"] == 0.5
    assert rows[0]["ema_dev_metric"] == 0.45
    assert rows[0]["optimizer_step"] == 781


def test_segment_acceptance_checks_streaming_hashes_without_loading_checkpoint(tmp_path: Path):
    root = tmp_path / "remote"
    training = root / "training"
    training.mkdir(parents=True)
    preflight = root / "preflight"
    preflight.mkdir()
    checkpoint = training / "last_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    (training / "checkpoint_epoch_09.pt").write_bytes(checkpoint.read_bytes())
    (training / "best_model.pt").write_bytes(b"best")
    (training / "development_predictions.pt").write_bytes(b"predictions")
    recorder = baseline._v5_audit_recorder(training / "canonical_trace.json")
    rows = []
    for epoch in range(10):
        row = {
            "epoch": epoch,
            "train_mae_eV": 0.4,
            "live_development_mae_eV": 0.5,
            "development_mae_eV": 0.45,
            "learning_rate": 0.00025,
            "elapsed_seconds": 12.0,
            "cumulative_wall_time_seconds": (epoch + 1) * 12.0,
            "cumulative_optimizer_steps": (epoch + 1) * 781,
            "cumulative_sample_presentations": (epoch + 1) * 99_968,
        }
        rows.append(row)
        identity = sha256_file(checkpoint) if epoch == 9 else f"{epoch:064x}"
        recorder.checkpoint_event(
            f"sha256:{identity}", optimizer_step=row["cumulative_optimizer_steps"],
            sample_presentations=row["cumulative_sample_presentations"],
            epoch_or_pass=epoch + 1, learning_rate=row["learning_rate"],
            live_train_metric=0.4, live_dev_metric=0.5, ema_dev_metric=0.45,
            wall_time_seconds=12.0, cumulative_wall_time_seconds=(epoch + 1) * 12.0,
        )
    atomic_json(training / "trace.json", {"format": baseline.RUN_FORMAT, "rows": rows})
    resume_hashes = {name: sha256_file(training / name) for name in RESUME_FILES}
    atomic_json(training / "partial_manifest.json", {
        "format": baseline.RUN_FORMAT, "v5_audit": True, "complete": False,
        "completed_epochs": 10, "next_epoch": 10,
        "checkpoint_sha256": sha256_file(checkpoint),
        "canonical_trace_sha256": sha256_file(training / "canonical_trace.json"),
        "manifest_sha256": baseline.MANIFEST_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "runtime_certificate_id": "certificate-test", "resume_file_sha256": resume_hashes,
    })
    atomic_json(preflight / "preflight.json", {
        "accepted": True, "variant": "reference", "parameters": baseline.EXPECTED_PARAMETERS,
        "manifest_sha256": baseline.MANIFEST_SHA256,
        "source_archive_sha256": SOURCE_SHA256, "source_commit": SOURCE_COMMIT,
        "runtime_certificate_id": "certificate-test",
        "runtime_certificate": {
            "status": "accepted", "physical_batch_per_device": 128,
            "precision": "fp32", "tf32_enabled": False, "accelerator": "Tesla T4",
        },
    })
    atomic_json(root / "native_cost.json", {
        "source_archive_sha256": SOURCE_SHA256, "source_commit": SOURCE_COMMIT,
        "completed_epochs": 10, "allocated_gpu_inventory": ["GPU 0: Tesla T4"],
        "allocated_gpu_count": 1, "used_gpu_count": 1, "wall_seconds": 150.0,
    })
    assert recorder.record["trajectory_id"] == TRAJECTORY
    assert recorder.record["run_id"] == RUN
    accepted = accept_segment(root, 10, tmp_path / "acceptance.json")
    assert accepted["accepted"] and accepted["next_segment_authorized"]
    checkpoint.write_bytes(b"corrupted")
    import pytest
    with pytest.raises(RuntimeError, match="Resume hash last_checkpoint.pt"):
        accept_segment(root, 10, tmp_path / "rejected.json")
