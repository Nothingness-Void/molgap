import hashlib
import json

import pandas as pd
import torch

from molgap.repaired_2m_3d_colab import (
    GRAPH_BUILD_CHUNKSIZE,
    GRAPH_BUILD_START_METHOD,
    LIGHT_SCHNET_CONFIG,
    ROUTE_B_SCHNET_CONFIG,
    _build_etkdg,
    _build_etkdg_parallel,
    build_secondary_graph_shard_from_array_manifest,
    prepare_secondary_array_inputs,
    split_graphs_by_source_idx,
)


def test_light_schnet_contract_matches_selected_compute_shape():
    assert LIGHT_SCHNET_CONFIG == {
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 6.0,
        "dropout": 0.0,
    }
    assert ROUTE_B_SCHNET_CONFIG == {
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
    }


def test_etkdg_builder_is_deterministic_and_preserves_labels():
    item = (17, "CCO", [-5.0, -1.0, 4.0], 42)
    source_a, graph_a = _build_etkdg(item)
    source_b, graph_b = _build_etkdg(item)

    assert source_a == source_b == 17
    assert graph_a is not None and graph_b is not None
    assert torch.equal(graph_a.source_idx, torch.tensor([17]))
    assert torch.equal(graph_a.y, torch.tensor([[-5.0, -1.0, 4.0]]))
    assert torch.allclose(graph_a.pos, graph_b.pos)
    assert torch.isfinite(graph_a.charges).all()


def test_parallel_etkdg_builder_uses_spawn_and_preserves_identity():
    assert GRAPH_BUILD_START_METHOD == "spawn"
    assert GRAPH_BUILD_CHUNKSIZE < 100
    work = [
        (17, "CCO", [-5.0, -1.0, 4.0], 42),
        (18, "CCN", [-5.1, -1.1, 4.0], 42),
    ]
    results = list(
        _build_etkdg_parallel(
            work,
            workers=2,
            progress_label="test",
        )
    )
    assert sorted(source_idx for source_idx, _ in results) == [17, 18]
    assert all(graph is not None for _, graph in results)


def test_independent_etkdg_seed_changes_view():
    first = _build_etkdg((17, "CCCO", [-5.0, -1.0, 4.0], 42))[1]
    second = _build_etkdg((17, "CCCO", [-5.0, -1.0, 4.0], 314159))[1]
    assert first is not None and second is not None
    assert not torch.allclose(first.pos, second.pos)


def test_source_aligned_split_filters_after_role_assignment():
    graphs = []
    for source_idx in range(10):
        graph = _build_etkdg((source_idx, "CCO", [-5.0, -1.0, 4.0], 42))[1]
        assert graph is not None
        graphs.append(graph)
    train, validation, test = split_graphs_by_source_idx(
        graphs,
        source_rows=10,
        seed=42,
    )
    roles = {}
    for role, part in enumerate((train, validation, test)):
        roles.update(
            (int(graph.source_idx.view(-1)[0]), role) for graph in part
        )
    permutation = torch.from_numpy(
        __import__("numpy").random.RandomState(42).permutation(10)
    )
    expected = torch.full((10,), 2, dtype=torch.long)
    expected[permutation[:8]] = 0
    expected[permutation[8:9]] = 1
    assert roles == {index: int(expected[index]) for index in roles}


def test_secondary_array_shard_is_hash_bound_and_resumable(tmp_path, monkeypatch):
    repaired_csv = tmp_path / "repaired.csv"
    rows = pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "homo": [-5.0, -5.1, -5.2, -5.3],
            "lumo": [-1.0, -1.1, -1.2, -1.3],
            "gap": [4.0, 4.0, 4.0, 4.0],
        }
    )
    rows.to_csv(repaired_csv, index=False)
    source_sha = hashlib.sha256(repaired_csv.read_bytes()).hexdigest()

    metadata_root = tmp_path / "metadata"
    reports_dir = metadata_root / "reports"
    reports_dir.mkdir(parents=True)
    accepted_shards = []
    for index, (start, stop, failures) in enumerate(
        ((0, 2, [1]), (2, 4, []))
    ):
        graph_name = f"graphs_{start:07d}_{stop:07d}.pt"
        graph_sha = f"{index + 1:064x}"
        sidecar = {
            "status": "complete",
            "start": start,
            "stop": stop,
            "failed": len(failures),
            "failure_source_idx": failures,
            "graphs": stop - start - len(failures),
            "sha256": graph_sha,
        }
        sidecar_path = reports_dir / graph_name.replace(".pt", ".json")
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        accepted_shards.append(
            {
                "path": graph_name,
                "rows": sidecar["graphs"],
                "sha256": graph_sha,
                "sidecar_path": f"reports/{sidecar_path.name}",
                "sidecar_sha256": hashlib.sha256(
                    sidecar_path.read_bytes()
                ).hexdigest(),
            }
        )
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "immutable": True,
                "expected_shards": 2,
                "source_csv_sha256": source_sha,
                "shards": accepted_shards,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "molgap.repaired_2m_3d_colab.REPAIRED_2M_SHA256", source_sha
    )
    monkeypatch.setattr(
        "molgap.repaired_2m_3d_colab._build_etkdg_parallel",
        lambda work, **_: (
            (source_idx, Data_for_test(source_idx, target))
            for source_idx, _, target, _ in work
        ),
    )
    array_root = tmp_path / "array"
    manifest = prepare_secondary_array_inputs(
        repaired_csv=repaired_csv,
        primary_acceptance=acceptance_path,
        primary_metadata_root=metadata_root,
        output_dir=array_root,
        start_shard=0,
        stop_shard=2,
        expected_shards=2,
        expected_source_rows=4,
    )
    assert len(manifest["shards"]) == 2
    assert manifest["shards"][0]["primary_failure_source_idx"] == [1]

    output_dir = tmp_path / "output"
    first = build_secondary_graph_shard_from_array_manifest(
        manifest_path=array_root / "manifest.json",
        output_dir=output_dir,
        shard_index=0,
        workers=1,
    )
    second = build_secondary_graph_shard_from_array_manifest(
        manifest_path=array_root / "manifest.json",
        output_dir=output_dir,
        shard_index=0,
        workers=1,
    )
    assert first == second
    assert first["requested"] == first["graphs"] == 1
    graph = torch.load(
        output_dir / "graphs_0000000_0000002.pt",
        map_location="cpu",
        weights_only=False,
    )[0]
    assert int(graph.source_idx.item()) == 0


def Data_for_test(source_idx, target):
    from torch_geometric.data import Data

    return Data(
        z=torch.tensor([6], dtype=torch.long),
        pos=torch.zeros((1, 3), dtype=torch.float32),
        charges=torch.zeros(1, dtype=torch.float32),
        y=torch.tensor(target, dtype=torch.float32).view(1, 3),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )
