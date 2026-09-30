"""Synthetic event-hook checks for the GPTrans V4 trainer.

These tests exercise the trainer boundary with metadata-only model state. They
do not construct GPTrans or run a model forward/training operation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import experiment_family_workflow as family_workflow
from molgap import pcqm_gptrans_v4 as core


class _Batch:
    def __init__(self, count: int):
        self.num_graphs = count

    def to(self, *_args, **_kwargs):
        return self


class _Model:
    def __init__(self):
        self.weight = torch.tensor([11.0])

    def train(self):
        pass

    def parameters(self):
        return [SimpleNamespace(numel=lambda: 1)]

    def state_dict(self):
        return {"weight": self.weight.clone()}

    def load_state_dict(self, _state, strict=True):
        assert strict


class _State:
    def __init__(self, name: str):
        self.name = name

    def state_dict(self):
        return {f"{self.name}_state": 1}

    def load_state_dict(self, _state):
        pass


class _EMA(_State):
    def __init__(self):
        super().__init__("ema")

    def state_dict(self):
        return {"weight": torch.tensor([22.0])}

    def update(self, _model):
        pass


class _Scheduler(_State):
    def __init__(self):
        super().__init__("scheduler")

    def step(self, _epoch):
        return 0.01


class _Recorder:
    def __init__(self, observations=()):
        self.record = {"observations": list(observations)}
        self.resume_events = []
        self.checkpoint_events = []

    def resume_event(self, *args, **kwargs):
        self.resume_events.append((args, kwargs))

    def checkpoint_event(self, *args, **kwargs):
        self.checkpoint_events.append((args, kwargs))


class _Session:
    def __init__(self, observations=()):
        self.stage = SimpleNamespace(recorder=_Recorder(observations))
        self.epochs = []
        self.selected_states = []
        self.checkpoints = []

    def epoch_finished(self, **fields):
        self.epochs.append(fields)
        self.stage.recorder.record["observations"].append(
            {"event": "observation", **fields}
        )

    def selected(self, **fields):
        self.selected_states.append(fields)

    def checkpoint(self, **fields):
        self.checkpoints.append(fields)


def _development(mae: float, marker: float):
    return {
        "mae_eV": mae,
        "prediction_eV": torch.tensor([marker, marker + 1]),
        "target_eV": torch.tensor([marker - 0.1, marker + 0.9]),
        "source_idx": torch.tensor([50_001, 50_000]),
    }


def _install_synthetic_runtime(monkeypatch, tmp_path: Path, *, epochs=2, evaluations=None):
    train_rows, batch_size = 4, 2
    batches_per_epoch = train_rows // batch_size
    development_rows = 2
    monkeypatch.setattr(core, "EPOCHS", epochs)
    monkeypatch.setattr(core, "TRAIN_ROWS", train_rows)
    monkeypatch.setattr(core, "PHYSICAL_BATCH", batch_size)
    monkeypatch.setattr(core, "BATCHES_PER_EPOCH", batches_per_epoch)
    monkeypatch.setattr(core, "DEVELOPMENT_ROWS", development_rows)
    monkeypatch.setattr(core, "TAIL_ROWS_PER_EPOCH", 0)
    monkeypatch.setattr(core, "SAMPLE_PRESENTATIONS", train_rows * epochs)

    # Run the existing trainer orchestration on CPU tensors while it believes
    # its hardware gate has been satisfied. All data/model operations below
    # are test doubles.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    tensor = torch.tensor
    zeros = torch.zeros

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)

    model, optimizer, scheduler, ema = _Model(), _State("optimizer"), _Scheduler(), _EMA()
    monkeypatch.setattr(core, "configure_fp32_determinism", lambda _seed: {"fixture": True})
    monkeypatch.setattr(core, "validate_source_archive", lambda *_args: "fixture-source")
    monkeypatch.setattr(
        core,
        "validate_fixed_assets",
        lambda *_args, **_kwargs: SimpleNamespace(
            train_paths=("train",), development_paths=("development",)
        ),
    )
    monkeypatch.setattr(
        core,
        "_load_datasets",
        lambda paths: (
            list(range(train_rows if paths == ("train",) else development_rows)),
            [object()] if paths == ("train",) else [],
        ),
    )
    monkeypatch.setattr(core, "_target_stats", lambda _shards: (0.0, 1.0))
    monkeypatch.setattr(core, "build_runtime_manifest", lambda _determinism: {"runtime_fingerprint": "fixture-runtime"})
    monkeypatch.setattr(core, "validate_runtime_certificate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        core,
        "_make_training_state",
        lambda *_args, **_kwargs: (model, optimizer, scheduler, ema),
    )
    monkeypatch.setattr(
        core,
        "_training_loader",
        lambda _graphs, _epoch: [_Batch(batch_size) for _ in range(batches_per_epoch)],
    )
    monkeypatch.setattr(
        core,
        "_optimizer_step",
        lambda *_args, **_kwargs: torch.tensor(2.0),
    )
    eval_values = list(evaluations or [_development(0.5, 10.0) for _ in range(epochs)])
    eval_index = iter(eval_values)
    monkeypatch.setattr(
        core,
        "_evaluate",
        lambda *_args, **_kwargs: next(eval_index),
    )
    rng_fixture = {"python": "synthetic-rng-state"}
    monkeypatch.setattr(core, "capture_rng_state", lambda: rng_fixture)
    monkeypatch.setattr(core, "restore_rng_state", lambda _state: None)
    monkeypatch.setattr(family_workflow, "tensor_safe_rng_state", lambda state: state)

    certificate = {"runtime_fingerprint": "fixture-runtime", "accelerator": "synthetic"}
    preflight = {
        "training_addon": {},
        "accepted": True,
        "variant": "reference",
        "source_archive_sha256": "fixture-archive-sha",
        "source_commit": "fixture-commit",
        "runtime_certificate": certificate,
        "runtime_certificate_id": "fixture-certificate-id",
        "target_stats": {"mean_eV": 0.0, "sample_std_eV": 1.0},
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    output = tmp_path / "output"
    kwargs = {
        "dataset_root": tmp_path,
        "manifest_path": tmp_path / "manifest.json",
        "preflight_path": preflight_path,
        "source_archive": tmp_path / "source.tar.gz",
        "source_archive_sha256": "fixture-archive-sha",
        "source_commit": "fixture-commit",
        "output": output,
        "platform_id": "synthetic-platform",
        "initial_state_path": tmp_path / "initial.pt",
        "runtime_calibration_fingerprint": "fixture-calibration",
    }
    return kwargs, output, model, ema, rng_fixture


def _resume_checkpoint(*, epoch: int):
    return {
        "format": core.CHECKPOINT_FORMAT,
        "variant": "reference",
        "training_addon": {},
        "scientific_fields": core._scientific_fields(),
        "runtime_certificate_id": "fixture-certificate-id",
        "source_archive_sha256": "fixture-archive-sha",
        "model": {"weight": torch.tensor([11.0])},
        "optimizer": {"optimizer_state": 1},
        "scheduler": {"scheduler_state": 1},
        "ema": {"weight": torch.tensor([22.0])},
        "rng_state": {"fixture": True},
        "epoch": epoch,
        "trace": [{"epoch": index} for index in range(epoch + 1)],
        "best_development_mae_eV": 10.0,
        "best_epoch": epoch,
    }


def test_run_training_emits_one_based_cumulative_events_and_ema_selection(monkeypatch, tmp_path):
    evaluations = [_development(0.5, 10.0), _development(0.6, 20.0)]
    kwargs, _output, model, _ema, rng_fixture = _install_synthetic_runtime(
        monkeypatch, tmp_path, evaluations=evaluations
    )
    session = _Session()

    result = core.run_training(**kwargs, output_session=session)

    assert result["complete"] is True
    assert [event["epoch"] for event in session.epochs] == [1, 2]
    assert [event["optimizer_step"] for event in session.epochs] == [2, 4]
    assert [event["sample_presentations"] for event in session.epochs] == [4, 8]
    assert [event["live_train_metric"] for event in session.epochs] == [2.0, 2.0]
    assert [event["ema_dev_metric"] for event in session.epochs] == [0.5, 0.6]

    # The first best EMA state and its paired development rows are selected
    # together. A later non-improving epoch emits no replacement selection.
    assert len(session.selected_states) == 1
    selected = session.selected_states[0]
    assert selected["epoch"] == 1
    assert selected["optimizer_step"] == 2
    assert selected["weights"] == "ema"
    torch.testing.assert_close(selected["model_state"]["weight"], torch.tensor([22.0]))
    torch.testing.assert_close(selected["prediction_eV"], evaluations[0]["prediction_eV"])
    torch.testing.assert_close(selected["target_eV"], evaluations[0]["target_eV"])
    torch.testing.assert_close(selected["source_idx"], evaluations[0]["source_idx"])

    assert len(session.checkpoints) == 2
    checkpoint = session.checkpoints[0]
    torch.testing.assert_close(checkpoint["model_state"]["weight"], model.weight)
    torch.testing.assert_close(checkpoint["ema_state"]["weight"], torch.tensor([22.0]))
    assert checkpoint["optimizer_step"] == 2
    assert checkpoint["sample_presentations"] == 4
    assert checkpoint["rng_state"] == rng_fixture
    assert checkpoint["cursor"]["epoch"] == 1
    assert checkpoint["cursor"]["next_batch"] == 0
    expected_order = torch.tensor(
        [index for ids in core.DeterministicEpochBatchSampler(core.TRAIN_ROWS, 0) for index in ids],
        dtype=torch.int64,
    )
    assert checkpoint["cursor"]["sampler_order_sha256"] == family_workflow.tensor_digest(
        expected_order, role="source_idx"
    )
    assert len(session.stage.recorder.checkpoint_events) == 2


def test_resume_reconciles_acknowledged_trace_count_and_cursor(monkeypatch, tmp_path):
    kwargs, output, _model, _ema, _rng = _install_synthetic_runtime(
        monkeypatch, tmp_path, epochs=3, evaluations=[_development(0.4, 30.0)]
    )
    checkpoint = _resume_checkpoint(epoch=1)
    (output / "last_checkpoint.pt").parent.mkdir(parents=True, exist_ok=True)
    (output / "last_checkpoint.pt").write_bytes(b"synthetic-checkpoint")
    checkpoint_sha256 = core.sha256_file(output / "last_checkpoint.pt")
    monkeypatch.setattr(core, "torch_load_compat", lambda *_args, **_kwargs: checkpoint)
    session = _Session([{"event": "observation"}, {"event": "observation"}])

    result = core.run_training(**kwargs, output_session=session)

    assert result["complete"] is True
    assert session.stage.recorder.resume_events == [
        ((checkpoint_sha256,), {
            "optimizer_step": 4,
            "sample_presentations": 8,
        })
    ]
    assert len(session.epochs) == 1
    assert session.epochs[0]["epoch"] == 3
    assert session.epochs[0]["optimizer_step"] == 6
    assert session.epochs[0]["sample_presentations"] == 12


def test_resume_rejects_trace_checkpoint_observation_count_mismatch(monkeypatch, tmp_path):
    kwargs, output, _model, _ema, _rng = _install_synthetic_runtime(
        monkeypatch, tmp_path, epochs=3, evaluations=[_development(0.4, 30.0)]
    )
    checkpoint = _resume_checkpoint(epoch=1)
    (output / "last_checkpoint.pt").parent.mkdir(parents=True, exist_ok=True)
    (output / "last_checkpoint.pt").write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(core, "torch_load_compat", lambda *_args, **_kwargs: checkpoint)
    session = _Session([{"event": "observation"}])

    with pytest.raises(RuntimeError, match="Family trace and durable training checkpoint disagree"):
        core.run_training(**kwargs, output_session=session)

    assert session.stage.recorder.resume_events == []
