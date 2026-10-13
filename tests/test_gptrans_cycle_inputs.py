"""Synthetic graph checks only; no molecular cache or model is executed."""
import itertools
import random

import networkx as nx
import pytest

from molgap.gptrans_cycle_inputs import adjacency, chordless_cycles, common_cycle_pairs


def test_all_four_node_graphs_against_independent_enumerator():
    possible = list(itertools.combinations(range(4), 2))
    for bits in range(2 ** len(possible)):
        edges = [edge for i, edge in enumerate(possible) if bits & (1 << i)]
        common_cycle_pairs(4, edges, independent_check=True)


@pytest.mark.parametrize("edges", [[], [(0, 1)],
    [(0, 1), (1, 2), (2, 0)],
    [(0, 1), (1, 2), (2, 3), (3, 0)],
    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
    [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
    [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 3)],
])
def test_empty_tree_chorded_fused_and_spiro(edges):
    result = common_cycle_pairs(7, edges, independent_check=True)
    neighbors = adjacency(7, edges)
    assert all(i != j and category == (1 if j in neighbors[i] else 2) for i, j, category in result)
    assert {(j, i, c) for i, j, c in result} == set(result)


def test_cap_and_chord_are_not_sssr_or_ring_membership():
    assert chordless_cycles(adjacency(19, list(nx.cycle_graph(19).edges()))) == ()
    square = common_cycle_pairs(4, list(nx.cycle_graph(4).edges()))
    assert (0, 2, 2) in square
    chorded = common_cycle_pairs(4, list(nx.cycle_graph(4).edges()) + [(0, 2)])
    assert (1, 3, 2) not in chorded


def test_permutation_and_edge_order_equivariance():
    rng = random.Random(42)
    for _ in range(30):
        nodes = 9
        edges = [e for e in itertools.combinations(range(nodes), 2) if rng.random() < .3]
        original = common_cycle_pairs(nodes, edges, independent_check=True)
        permutation = list(range(nodes))
        rng.shuffle(permutation)
        changed = [(permutation[i], permutation[j]) for i, j in edges]
        changed += [(j, i) for i, j in changed]
        rng.shuffle(changed)
        observed = common_cycle_pairs(nodes, changed, independent_check=True)
        assert set(observed) == {(permutation[i], permutation[j], c) for i, j, c in original}


@pytest.mark.parametrize("edges", [[(0, 0)], [(0, 3)], [(-1, 0)]])
def test_invalid_edges_fail_closed(edges):
    with pytest.raises(ValueError):
        common_cycle_pairs(3, edges)


def test_chunk_validation_and_pyg_offsets():
    import torch
    from torch_geometric.data import Data, Batch
    from molgap.gptrans_cycle_inputs import CycleDataset, validate_chunk
    pairs = common_cycle_pairs(3, [(0, 1), (1, 2), (2, 0)])
    payload = {"source_idx": torch.tensor([8, 9]), "nodes": torch.tensor([3, 2]),
        "ptr": torch.tensor([0, 6, 6]),
        "pair_index": torch.tensor([(i, j) for i, j, _ in pairs]).t().contiguous(),
        "category": torch.tensor([c for _, _, c in pairs], dtype=torch.uint8)}
    record = {"source_start": 8, "source_stop": 10, "rows": 2, "pairs": 6}
    validate_chunk(payload, record)
    base = [Data(x=torch.zeros(n, 9, dtype=torch.long), source_idx=torch.tensor([idx]))
            for idx, n in ((8, 3), (9, 2))]
    wrapped = CycleDataset(base, [(8, 10, payload)])
    assert wrapped[1].cycle_pair_index.shape == (2, 0)
    batch = Batch.from_data_list([wrapped[0], wrapped[1], wrapped[0]])
    assert torch.equal(batch.cycle_pair_index[:, 6:], payload["pair_index"] + 5)
    assert not hasattr(base[0], "cycle_pair_index")
    invalid = dict(payload, category=torch.zeros(6, dtype=torch.uint8))
    with pytest.raises(ValueError):
        validate_chunk(invalid, record)
    invalid = dict(payload, ptr=torch.tensor([1, 6, 6]))
    with pytest.raises(ValueError):
        validate_chunk(invalid, record)
