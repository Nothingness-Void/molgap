import hashlib
import pickle

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap import k1_screen_training as owner
from molgap.k1_local_speed import (
    EPOCH_FORMAT, summarize, summarize_epochs, timed_train_epoch,
    validate_config, verify_source,
)


def config():
    return {"format": "molgap-k1-local-speed-v1", "epochs": 10,
        "allocation_seconds": 3600, "schedule_epochs": 40,
        "batch_size": 128, "development_access": False, "precision": "fp32-tf32-off"}


@pytest.mark.parametrize("key,value", [("epochs", 11), ("schedule_epochs", 10),
    ("allocation_seconds", 7200), ("development_access", True), ("precision", "tf32")])
def test_contract_rejects_scope_expansion(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_engineering_mode_has_no_rml_authority():
    c = {**config(), "format": EPOCH_FORMAT, "rml_recording": False}
    assert validate_config(c) == c
    for changed in ({"rml_recording": True}, {"prospective": {}}):
        with pytest.raises(ValueError):
            validate_config({**c, **changed})


def test_epoch_summary_checks_matched_exposure():
    rows = [{"arm": arm, "epoch": epoch, "steps": owner.STEPS_PER_EPOCH,
             "samples": owner.ROWS_PER_EPOCH, "train_seconds": seconds}
            for epoch in range(1, 11)
            for arm, seconds in (("reference", 10.0), ("fused_layout", 8.0))]
    result = summarize_epochs(rows)
    assert result["reference"]["steps"] == 7810
    assert result["time_reduction_fraction"] == pytest.approx(0.2)
    assert result["rml_recording"] is False
    with pytest.raises(ValueError):
        summarize_epochs(rows[:-1])
    rows[0]["train_seconds"] = float("nan")
    with pytest.raises(ValueError):
        summarize_epochs(rows)


@pytest.mark.parametrize("arm", ["reference", "fused_layout"])
def test_epoch_adapter_reuses_train_step_without_per_batch_sync(monkeypatch, arm):
    from types import SimpleNamespace
    from molgap import k1_local_speed as speed

    calls, syncs = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: syncs.append(1))
    monkeypatch.setattr(owner, "STEPS_PER_EPOCH", 2)
    monkeypatch.setattr(owner, "ROWS_PER_EPOCH", 256)
    monkeypatch.setattr(owner, "_train_loader", lambda train, epoch: [1, 2])
    def step(model, optimizer, batch, mean, std, mode, **kwargs):
        calls.append((mode, kwargs))
        return torch.tensor(1.0), torch.tensor(128.0), 128
    monkeypatch.setattr(owner, "_screen_step", step)
    row = timed_train_epoch(SimpleNamespace(train=lambda: None),
        SimpleNamespace(param_groups=[{"lr": 4e-4}]), [], 0.0, 1.0, arm, 0)
    assert row["steps"] == 2
    assert len(syncs) == 2
    assert calls == [(arm, {"original_execution": arm == "reference"})] * 2


def test_source_inventory_cannot_redefine_its_own_authority(tmp_path):
    import json

    inventory = tmp_path / "SOURCE_FILES.json"
    inventory.write_text(json.dumps({"source_commit": "a" * 40, "files": []}))
    expected = owner.sha256_file(inventory)
    inventory.write_text(json.dumps({"source_commit": "a" * 40, "files": [{"path": "injected.py", "sha256": "b" * 64}]}))
    with pytest.raises(ValueError, match="authority changed"):
        verify_source(tmp_path, {"source_inventory_sha256": expected})


def test_train_only_never_opens_development(tmp_path, monkeypatch):
    data, slices = InMemoryDataset.collate([
        Data(x=torch.zeros(2, 9), source_idx=torch.tensor([i]), y=torch.tensor([1.0]),
             pos=torch.zeros(2, 3)) for i in range(4)])
    torch.save((data, slices), tmp_path / "train.pt")
    digest = owner.sha256_file(tmp_path / "train.pt")
    shards = [{"file": "train.pt", "role": "train", "rows": 4, "sha256": digest},
        {"file": "DEVELOPMENT_MUST_NOT_EXIST.pt", "role": "development", "rows": 2, "sha256": "a" * 64}]
    aggregate = hashlib.sha256()
    for row in shards:
        aggregate.update(f"{row['role']}\tstore/geometry/{row['file']}\t{row['sha256']}\n".encode())
    monkeypatch.setattr(owner, "FIXED_GEOMETRY_SHA256", aggregate.hexdigest())
    monkeypatch.setattr(owner, "TRAIN_ROWS", 4)
    roles = owner.load_roles(tmp_path, {"geometry_shards": shards}, selected_roles=("train",))
    assert set(roles) == {"train"}
    assert len(roles["train"]) == 4
    assert "pos" not in roles["train"][0]
    restored = pickle.loads(pickle.dumps(roles["train"]))
    assert torch.equal(restored[3].source_idx, torch.tensor([3]))
    with pytest.raises(FileNotFoundError):
        owner.load_roles(tmp_path, {"geometry_shards": shards})


def test_summary_retains_all_exposure_but_excludes_only_steady_warmup():
    rows = [{"arm": arm, "batch": i, "samples": 128, "step_seconds": seconds,
             "pipeline_seconds": seconds + 0.01} for i in range(10)
            for arm, seconds in (("reference", 0.1), ("fused_layout", 0.07))]
    result = summarize(rows)
    assert result["reference"]["samples"] == 1280
    assert result["fused_layout"]["steady_step_median_seconds"] == 0.07
    assert result["quality_evaluated"] is False
    with pytest.raises(ValueError):
        summarize(rows[:-1])


def test_saved_speed_acceptance_rejects_missing_steps_and_nonfinite_timing():
    import importlib.util
    import math
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "experiments/pcqm_k1_fused_layout_speed/accept.py"
    spec = importlib.util.spec_from_file_location("local_speed_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"arm": arm, "epoch": e, "batch": b, "samples": 128,
        "step_seconds": 0.1, "pipeline_seconds": 0.11,
        "learning_rate": 1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * e / 40)) / 2}
        for e in range(10) for b in range(781) for arm in ("reference", "fused_layout")]
    assert module.inspect_rows(rows)
    with pytest.raises(ValueError):
        module.inspect_rows(rows[:-1])
    rows[0]["step_seconds"] = float("nan")
    with pytest.raises(ValueError):
        module.inspect_rows(rows)
    rows[0]["step_seconds"] = 0.1
    rows[-1]["learning_rate"] = 1e-6
    with pytest.raises(ValueError):
        module.inspect_rows(rows)
