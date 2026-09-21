from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.k1_pair_token import MODE as PAIR_TOKEN_MODE, make_encoder as make_pair_token
from molgap.k1_pair_token_mose import MODE, PARAMETERS, check_mechanism, make_encoder
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder as make_variant


ROOT = Path(__file__).resolve().parent


def _batch():
    graph = Data(
        x=torch.zeros((4, 9), dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
        edge_attr=torch.zeros((6, 3), dtype=torch.long),
        random_walk_pe=torch.cat(
            (torch.zeros((4, 16)), torch.arange(124).reshape(4, 31).float().log1p()),
            dim=-1,
        ),
    )
    return Batch.from_data_list([graph])


def test_contract_and_registry():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    config = ARCHITECTURE_CONFIGS[MODE]
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["mose_aggregate_sha256"] == "5d949f90a35aea5001d2f6438c916f92cab45d6c5860ba6ccf5777aac1c897e3"
    assert config["expected_parameters"] == PARAMETERS[MODE]
    assert config["prediction_fusion"] is False


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
        base_args = (batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe[:, :16])
        base_output = baseline(*base_args)
        pair_output = pair_token(*base_args)
        candidate_output = candidate(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe
        )
    assert torch.equal(base_output, pair_output)
    assert torch.equal(base_output, candidate_output)
    assert torch.count_nonzero(candidate.mose_residual[-1].weight) == 0
    assert torch.count_nonzero(candidate.relation_token.return_projection.weight) == 0


def test_mechanism_invariants():
    candidate = make_encoder(MODE)
    checks = check_mechanism(candidate, _batch())
    for name in (
        "valid_pair_count_exact",
        "assignment_mass_one",
        "padding_mass_zero",
        "zero_return_projection",
        "zero_initial_update_exact",
        "mose_input_finite",
        "mose_input_nonnegative",
        "zero_mose_return",
        "zero_initial_mose_update_exact",
        "pairtoken_parent_unchanged",
    ):
        assert checks[name] is True
    assert checks["combined_input_shape"] == [4, 47]
    assert checks["prediction_fusion"] is False


def test_kaggle3_binding():
    metadata = json.loads((ROOT / "kaggle_candidate/kernel-metadata.json").read_text())
    assert metadata["id"] == "nvoid912/molgap-k1-pairtoken-mose-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "nvoid912/molgap-pcqm-k1-mose-cache-v1" in metadata["dataset_sources"]
