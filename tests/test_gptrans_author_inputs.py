"""Focused synthetic checks for the bounded GPTrans CPU input APIs."""

import json

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import molgap.gptrans_author_inputs as author_inputs
from molgap.gptrans_author_inputs import (
    PathData,
    graph_paths,
    path_histogram,
    path_mean_from_bond_tables,
    prepare_degree_initial_state,
)


def _bidirectional(edges):
    result = []
    for source, target, category in edges:
        result.extend(
            [(source, target, category), (target, source, category)]
        )
    return result


def test_graph_paths_is_sorted_bfs_directed_and_capped():
    edges = _bidirectional(
        [
            (0, 1, (0, 0, 0)),
            (0, 2, (1, 1, 1)),
            (1, 3, (2, 2, 1)),
            (2, 3, (3, 3, 0)),
        ]
    )
    result = graph_paths(4, edges)

    # The direct bond pairs are absent.  The 0->3 tie uses the sorted 0->1
    # neighbor first, so its path categories are reproducible.
    assert result["path_pair_src"].tolist() == [0, 1, 2, 3]
    assert result["path_pair_dst"].tolist() == [3, 2, 1, 0]
    assert result["path_pair_distance"].tolist() == [2, 2, 2, 2]
    assert result["path_edge_attr"][:2].tolist() == [[0, 0, 0], [2, 2, 1]]
    assert torch.equal(
        result["path_pair_hist"].long(),
        path_histogram(result["path_pair_ptr"], result["path_edge_attr"]),
    )

    chain = _bidirectional(
        [(index, index + 1, (0, 0, 0)) for index in range(22)]
    )
    capped = graph_paths(23, chain)
    pairs = set(zip(capped["path_pair_src"].tolist(), capped["path_pair_dst"].tolist()))
    assert (0, 20) in pairs
    assert (0, 21) not in pairs
    assert all(distance >= 2 for distance in capped["path_pair_distance"].tolist())


def test_path_mean_uses_only_existing_bond_tables_and_divides_by_length():
    result = graph_paths(
        3,
        _bidirectional(
            [
                (0, 1, (1, 2, 1)),
                (1, 2, (3, 4, 0)),
            ]
        ),
    )
    tables = [
        torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4),
        torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4),
        torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4),
    ]
    observed = path_mean_from_bond_tables(
        result["path_pair_distance"], result["path_pair_hist"], tables
    )
    expected = (tables[0][1] + tables[0][3] + tables[1][2] + tables[1][4]
                + tables[2][1] + tables[2][0]) / 2
    assert observed.shape == (2, 4)
    assert torch.allclose(observed[0], expected)
    assert torch.allclose(observed[1], expected)


def test_path_data_collates_pair_indices_by_num_nodes():
    first = PathData(
        x=torch.zeros((3, 1)),
        path_pair_index=torch.tensor([[0, 1], [2, 0]]),
        path_counts=torch.ones((2, 13), dtype=torch.int16),
        path_lengths=torch.tensor([2, 3]),
    )
    second = PathData(
        x=torch.zeros((2, 1)),
        path_pair_index=torch.tensor([[0], [1]]),
        path_counts=torch.ones((1, 13), dtype=torch.int16),
        path_lengths=torch.tensor([2]),
    )
    batch = next(iter(DataLoader([first, second], batch_size=2)))
    assert batch.path_pair_index.tolist() == [[0, 1, 3], [2, 0, 4]]
    assert batch.path_counts.shape == (3, 13)
    assert batch.path_lengths.tolist() == [2, 3, 2]


def test_bounded_chunk_payload_rederives_without_full_shard_accumulation():
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [2, 2, 1], [2, 2, 1]]
    )
    graphs = [
        Data(
            x=torch.zeros((4, 9), dtype=torch.long),
            edge_index=edges,
            edge_attr=edge_attr,
            source_idx=torch.tensor([index]),
        )
        for index in range(1001)
    ]
    record = {"rows": 1000, "source_idx_min": 0, "source_idx_max": 999}
    payload = author_inputs._build_shard_payload(
        graphs, record, include_path_hist=True, start=0, end=1000
    )
    assert payload["source_index"].numel() == 1000
    assert int(payload["source_index"][-1]) == 999
    assert author_inputs._validate_payload(
        payload, record, include_path_hist=True
    )["rows"] == 1000
    author_inputs._rederive_shard(
        payload, graphs, {"source_idx_min": 0}, include_path_hist=True,
        start=0, end=1000,
    )


def test_path_dataset_loads_compact_chunks_once_and_caps_cache(tmp_path, monkeypatch):
    records = []
    graphs = []
    for source_index in (0, 1):
        payload = {
            "include_path_hist": True,
            "source_index": torch.tensor([source_index]),
            "graph_pair_ptr": torch.tensor([0, 1]),
            "path_pair_src": torch.tensor([0]),
            "path_pair_dst": torch.tensor([1]),
            "path_pair_distance": torch.tensor([2]),
            "path_pair_hist": torch.tensor(
                [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int16,
            ),
            # These verification-only tensors must not survive in PathDataset.
            "path_pair_ptr": torch.tensor([0, 2]),
            "path_edge_attr": torch.zeros((2, 3), dtype=torch.int16),
        }
        filename = f"part-{source_index}.pt"
        torch.save(payload, tmp_path / filename)
        records.append(
            {
                "file": filename,
                "rows": 1,
                "pairs": 1,
                "source_idx_min": source_index,
                "source_idx_max": source_index,
            }
        )
        graphs.append(
            Data(
                x=torch.zeros((2, 1)),
                source_idx=torch.tensor([source_index]),
            )
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"include_path_hist": True, "shards": records}),
        encoding="utf-8",
    )

    load_count = {"value": 0}
    original_load = author_inputs._torch_load

    def counted_load(path):
        load_count["value"] += 1
        return original_load(path)

    monkeypatch.setattr(author_inputs, "_torch_load", counted_load)
    estimate = author_inputs._runtime_compact_cache_estimate(records)
    dataset = author_inputs.PathDataset(
        graphs,
        tmp_path,
        {"runtime_compact_cache": estimate},
    )
    assert load_count["value"] == 2
    assert all(set(payload) == set(author_inputs.RUNTIME_COMPACT_KEYS)
               for payload in dataset._compact_payloads)
    assert author_inputs.PATH_PAIR_PTR not in dataset._compact_payloads[0]
    assert author_inputs.PATH_EDGE_ATTR not in dataset._compact_payloads[0]
    assert dataset[1].path_counts.shape == (1, 13)
    assert load_count["value"] == 2

    monkeypatch.setattr(
        author_inputs,
        "RUNTIME_COMPACT_BYTES_CAP",
        estimate["estimated_bytes"] - 1,
    )
    with pytest.raises(RuntimeError, match="4 GiB in-memory cap"):
        author_inputs.PathDataset(graphs, tmp_path, {})


def test_prepare_degree_initial_state_scales_only_pinned_tables(tmp_path, monkeypatch):
    state = {
        "in_degree_encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "out_degree_encoder.weight": torch.full((3, 4), 2.0),
        "unrelated.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    input_path = tmp_path / "initial_state.pt"
    output_path = tmp_path / "degree_scaled.pt"
    torch.save(
        {
            "format": "molgap-gptrans-t-seed42-initial-state-v1",
            "model_state": state,
            "state_sha256": author_inputs.state_digest(state),
        },
        input_path,
    )
    monkeypatch.setattr(author_inputs, "INITIAL_STATE_SHA256", author_inputs.digest(input_path))
    monkeypatch.setattr(author_inputs, "INITIAL_MODEL_SHA256", author_inputs.state_digest(state))

    result = prepare_degree_initial_state(input_path, output_path)
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert json.dumps(result, sort_keys=True)
    assert result["base_input_sha256"] == author_inputs.digest(input_path)
    assert result["base_state_sha256"] == author_inputs.state_digest(state)
    assert payload["state_sha256"] == result["state_sha256"]
    assert torch.equal(
        payload["model_state"]["unrelated.weight"], state["unrelated.weight"]
    )
    for name in ("in_degree_encoder.weight", "out_degree_encoder.weight"):
        assert torch.equal(
            payload["model_state"][name], state[name] * author_inputs.DEGREE_SCALE
        )


def test_prepare_degree_initial_state_rejects_unpinned_input(tmp_path):
    input_path = tmp_path / "not_frozen.pt"
    torch.save({"format": "wrong"}, input_path)
    with pytest.raises(RuntimeError, match="artifact changed"):
        prepare_degree_initial_state(input_path, tmp_path / "out.pt")
