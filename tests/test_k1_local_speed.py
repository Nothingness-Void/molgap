import hashlib
import pickle

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap import k1_screen_training as owner
from molgap.k1_local_speed import summarize, validate_config


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
