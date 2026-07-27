import torch

from molgap.repaired_2m_3d_colab import (
    GRAPH_BUILD_CHUNKSIZE,
    GRAPH_BUILD_START_METHOD,
    LIGHT_SCHNET_CONFIG,
    ROUTE_B_SCHNET_CONFIG,
    _build_etkdg,
    _build_etkdg_parallel,
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
