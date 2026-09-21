from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.k1_oneshot_triplet_pair_token import (
    MODE,
    PARAMETERS,
    TARGET_LAYER,
    TRIPLET_CHANNELS,
    check_mechanism,
    make_encoder,
)
from molgap.k1_pair_token import MODE as PAIR_TOKEN_MODE, make_encoder as make_pair_token
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder as make_variant
from molgap.pcqm_wedge import directed_nonbacktracking_wedges


ROOT = Path(__file__).resolve().parent


def _batch():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    graph = Data(
        x=torch.zeros((4, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.zeros((6, 3), dtype=torch.long),
        random_walk_pe=torch.zeros((4, 16)),
        wedge_edge_ids=directed_nonbacktracking_wedges(edge_index),
    )
    return Batch.from_data_list([graph])


def test_contract_and_registry():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    config = ARCHITECTURE_CONFIGS[MODE]
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["total_optimizer_steps"] == 31_240
    assert config["target_layer"] == TARGET_LAYER
    assert config["triplet_channels"] == TRIPLET_CHANNELS
    assert config["expected_parameters"] == PARAMETERS[MODE]
    assert config["persistent_triplet_state"] is False
    assert config["direct_node_return"] is False


def test_parameter_identity_and_exact_nested_function():
    batch = _batch()
    torch.manual_seed(42)
    baseline = make_variant("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    pair_token = make_pair_token(PAIR_TOKEN_MODE).eval()
    torch.manual_seed(42)
    candidate = make_encoder(MODE).eval()
    assert sum(p.numel() for p in candidate.parameters()) == PARAMETERS[MODE]
    with torch.no_grad():
        args = (batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe)
        base_output = baseline(*args)
        pair_output = pair_token(*args)
        candidate_output = candidate(*args, batch.wedge_edge_ids)
    assert torch.equal(base_output, pair_output)
    assert torch.equal(base_output, candidate_output)
    assert torch.count_nonzero(candidate.triplet_adapter.return_projection.weight) == 0
    assert torch.count_nonzero(candidate.relation_token.return_projection.weight) == 0


def test_mechanism_invariants():
    candidate = make_encoder(MODE)
    checks = check_mechanism(candidate, _batch())
    for name in (
        "adjacency_exact",
        "non_backtracking_exact",
        "center_identity_exact",
        "wedge_count_exact",
        "wedge_set_exact",
        "message_targets_outgoing_edge",
        "zero_triplet_return",
        "zero_initial_triplet_update_exact",
        "zero_pairtoken_return",
    ):
        assert checks[name] is True
    assert checks["persistent_triplet_state"] is False
    assert checks["direct_node_return"] is False
    assert checks["topology_only"] is True


def test_kaggle3_binding():
    metadata = json.loads((ROOT / "kaggle_candidate/kernel-metadata.json").read_text())
    assert metadata["id"] == "nvoid912/molgap-k1-one-shot-triplet-pairtoken-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
