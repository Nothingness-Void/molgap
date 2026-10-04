"""Mounted path resolution only: no graphs, weights or inference."""
import importlib.util
from pathlib import Path

import pytest


def bootstrap():
    source = Path(__file__).resolve().parents[1]/"experiments/pcqm_gptrans_ema_portability/run.py"
    spec = importlib.util.spec_from_file_location("frozen_audit_bootstrap", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duplicate_names_are_resolved_by_exact_dataset(tmp_path):
    paths = []
    for dataset in ("pcqm4mv2-ogb-fixed-100k-v1", "pcqm4mv2-ogb-fixed-500k-scnet-v1"):
        p=tmp_path/"datasets/kaseichou"/dataset/"train/train_shard_0002.pt"
        p.parent.mkdir(parents=True)
        p.touch()
        paths.append(p)
    resolver=bootstrap()
    with pytest.raises(ValueError):
        resolver.one("train_shard_0002.pt", input_root=tmp_path)
    assert resolver.one("train_shard_0002.pt", "pcqm4mv2-ogb-fixed-100k-v1", input_root=tmp_path)==paths[0]
    assert resolver.one("train_shard_0002.pt", "pcqm4mv2-ogb-fixed-500k-scnet-v1", input_root=tmp_path)==paths[1]


def test_missing_owning_mount_never_uses_another_dataset(tmp_path):
    p=tmp_path/"pcqm4mv2-ogb-fixed-500k-scnet-v1/train/train_shard_0002.pt"
    p.parent.mkdir(parents=True)
    p.touch()
    with pytest.raises(ValueError):
        bootstrap().one("train_shard_0002.pt", "pcqm4mv2-ogb-fixed-100k-v1", input_root=tmp_path)
