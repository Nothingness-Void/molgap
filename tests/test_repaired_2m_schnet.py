from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from molgap.repaired_2m_schnet import (
    Repaired2MSchNetConfig,
    _role_graphs,
    source_roles,
    stable_recovery_config,
    validate_evaluation_consistency,
    verify_accepted_graph_cache,
)


def _graph(source_idx: int) -> Data:
    return Data(
        source_idx=torch.tensor([source_idx]),
        y=torch.zeros(1, 3),
    )


def test_source_roles_are_complete_and_deterministic() -> None:
    first = source_roles(100, 42)
    second = source_roles(100, 42)
    assert np.array_equal(first, second)
    assert np.bincount(first, minlength=3).tolist() == [80, 10, 10]


def test_role_graphs_uses_source_aligned_split() -> None:
    roles = source_roles(20, 7)
    graphs = [_graph(index) for index in range(20)]
    for role in range(3):
        selected = _role_graphs(graphs, roles, role)
        assert all(roles[int(graph.source_idx)] == role for graph in selected)
    assert sum(len(_role_graphs(graphs, roles, role)) for role in range(3)) == 20


def test_role_graphs_rejects_out_of_range_source_idx() -> None:
    with pytest.raises(ValueError, match="outside split contract"):
        _role_graphs([_graph(10)], np.zeros(10, dtype=np.int8), 0)


def test_variant_contract() -> None:
    assert Repaired2MSchNetConfig("primary").variant == "primary"
    assert Repaired2MSchNetConfig("augmented").variant == "augmented"
    with pytest.raises(ValueError, match="variant"):
        Repaired2MSchNetConfig("secondary")


def test_stable_recovery_contract() -> None:
    primary = stable_recovery_config("primary")
    augmented = stable_recovery_config("augmented")
    assert primary.learning_rate == 5.0e-5
    assert augmented.learning_rate == 1.0e-4
    assert primary.grad_clip == augmented.grad_clip == 1.0
    assert primary.amp_dtype == augmented.amp_dtype == "bfloat16"
    assert primary.max_nonfinite_batches == 0


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_cache(tmp_path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    records = []
    for index in range(100):
        path = graph_dir / f"graphs_{index:07d}_{index + 1:07d}.pt"
        path.write_bytes(f"shard-{index}".encode("ascii"))
        records.append(
            {"path": path.name, "rows": 1, "sha256": _sha256(path)}
        )
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "accepted": True,
                "expected_shards": 100,
                "accepted_rows": 100,
                "source_rows": 100,
                "shards": records,
            }
        ),
        encoding="utf-8",
    )
    return graph_dir, acceptance


def test_verify_accepted_graph_cache_hashes_every_shard(tmp_path) -> None:
    graph_dir, acceptance = _accepted_cache(tmp_path)
    result = verify_accepted_graph_cache(graph_dir, acceptance)
    assert result["accepted_rows"] == 100
    assert result["shards"] == 100
    assert len(result["graph_ledger_sha256"]) == 64


def test_verify_accepted_graph_cache_rejects_changed_shard(tmp_path) -> None:
    graph_dir, acceptance = _accepted_cache(tmp_path)
    (graph_dir / "graphs_0000050_0000051.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash differs"):
        verify_accepted_graph_cache(graph_dir, acceptance)


def test_evaluation_consistency_accepts_reproducible_random_split() -> None:
    config = Repaired2MSchNetConfig("primary")
    report = validate_evaluation_consistency(
        validation_mae=np.array([0.12, 0.11, 0.16]),
        test_mae=np.array([0.121, 0.109, 0.159]),
        replay_test_mae=np.array([0.121, 0.109, 0.159]),
        config=config,
        expected_validation_average_mae_eV=0.13,
    )
    assert report["test_replay_max_delta_eV"] == 0.0


@pytest.mark.parametrize(
    ("replay", "expected", "message"),
    [
        (np.array([0.13, 0.11, 0.16]), 0.13, "replay differs"),
        (np.array([0.12, 0.11, 0.16]), 0.20, "source checkpoint"),
    ],
)
def test_evaluation_consistency_rejects_invalid_evidence(
    replay, expected, message
) -> None:
    config = Repaired2MSchNetConfig("primary")
    with pytest.raises(RuntimeError, match=message):
        validate_evaluation_consistency(
            validation_mae=np.array([0.12, 0.11, 0.16]),
            test_mae=np.array([0.12, 0.11, 0.16]),
            replay_test_mae=replay,
            config=config,
            expected_validation_average_mae_eV=expected,
        )


def test_evaluation_consistency_rejects_split_divergence() -> None:
    config = Repaired2MSchNetConfig("primary")
    with pytest.raises(RuntimeError, match="random-split contract"):
        validate_evaluation_consistency(
            validation_mae=np.array([0.12, 0.11, 0.16]),
            test_mae=np.array([0.22, 0.80, 0.81]),
            replay_test_mae=np.array([0.22, 0.80, 0.81]),
            config=config,
        )
