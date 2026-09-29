"""No-GPU checks for the additive GPTrans reference audit path."""

from pathlib import Path

import torch

from molgap import pcqm_gptrans_v4 as baseline
from molgap.research_memory.trace import load_canonical_trace


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
