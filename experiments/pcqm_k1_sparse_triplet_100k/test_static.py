from __future__ import annotations

import json
from pathlib import Path

import torch

from molgap.k1_sparse_triplet import MODE, PARAMETERS, check_mechanism, make_encoder
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder as make_variant
from molgap.pcqm_wedge import WedgeData, directed_nonbacktracking_wedges


ROOT = Path(__file__).resolve().parent


def test_frozen_contract_identity():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["arm"] == MODE
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["epochs"] == 40
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["total_sample_presentations"] == 3_998_720


def test_architecture_registry_is_topology_only():
    config = ARCHITECTURE_CONFIGS[MODE]
    assert config["triplet_channels"] == 16
    assert config["triplet_layers"] == 9
    assert config["expected_parameters"] == PARAMETERS[MODE]
    assert config["geometry"] is False
    assert config["external_features"] is False
    assert config["dense_atom_attention"] is False


def test_model_parameter_identity_and_zero_nested_initialization():
    torch.manual_seed(42)
    model = make_encoder(MODE)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[MODE]
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in model.triplet_to_edge)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in model.triplet_to_edge)
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in model.triplet_to_node)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in model.triplet_to_node)


def test_wedge_builder_is_complete_and_nonbacktracking():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]], dtype=torch.long
    )
    wedges = directed_nonbacktracking_wedges(edge_index)
    assert wedges.shape == (6, 2)
    first, second = wedges.unbind(dim=1)
    assert torch.equal(edge_index[1, first], edge_index[0, second])
    assert bool((edge_index[0, first] != edge_index[1, second]).all())
    assert torch.unique(wedges, dim=0).shape[0] == wedges.shape[0]


def test_batched_wedge_identity_and_exact_k1_initial_function():
    from torch_geometric.data import Batch

    def graph(nodes, edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return WedgeData(
            x=torch.zeros((nodes, 9), dtype=torch.long),
            edge_index=edge_index,
            edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
            random_walk_pe=torch.zeros((nodes, 16), dtype=torch.float32),
            wedge_edge_ids=directed_nonbacktracking_wedges(edge_index),
        )

    batch = Batch.from_data_list(
        [
            graph(3, [(0, 1), (1, 0), (1, 2), (2, 1)]),
            graph(4, [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1)]),
        ]
    )
    torch.manual_seed(42)
    baseline = make_variant("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    candidate = make_encoder(MODE).eval()
    with torch.no_grad():
        baseline_output = baseline(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
        candidate_output = candidate(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.wedge_edge_ids,
        )
    assert torch.equal(baseline_output, candidate_output)
    checks = check_mechanism(candidate, batch)
    assert checks["wedge_set_exact"] is True
    assert checks["wedge_pairs_unique"] is True


def test_kaggle3_single_t4_binding():
    metadata = json.loads((ROOT / "kaggle_candidate/kernel-metadata.json").read_text())
    assert metadata["id"] == "nvoid912/molgap-k1-sparse-triplet-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "nvoid912/molgap-k1-sparse-triplet-source",
        "nvoid912/pcqm4mv2-ogb-fixed-100k-v1",
    ]
