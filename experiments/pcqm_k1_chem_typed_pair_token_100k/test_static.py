from pathlib import Path
import json

import torch

from molgap.k1_chem_typed_pair_token import (
    ADDED_PARAMETERS,
    MODE,
    PARAMETERS,
    make_encoder,
)
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS


ROOT = Path(__file__).resolve().parent


def test_contract_and_registry_match() -> None:
    config = ARCHITECTURE_CONFIGS[MODE]
    assert config["added_parameters"] == ADDED_PARAMETERS == 23_600
    assert config["expected_parameters"] == PARAMETERS[MODE] == 3_682_417
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["physical_batch_per_device"] == 128
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["total_sample_presentations"] == 3_998_720
    assert contract["architecture"]["added_vs_pair_token"] == 752
    assert contract["materiality_rule"]["effective_gate_eV"] == 0.003


def test_model_parameter_and_nested_initialization_identity() -> None:
    model = make_encoder(MODE)
    token = model.relation_token
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[MODE]
    assert torch.count_nonzero(token.role_query).item() == 0
    assert torch.count_nonzero(token.return_projection.weight).item() == 0
    membership = torch.zeros((3, 12))
    roles = token._atom_roles(membership)
    assert torch.count_nonzero(roles).item() == 0


def test_original_pair_token_initialization_is_preserved() -> None:
    from molgap.k1_pair_token import MODE as PAIR_MODE, make_encoder as make_pair

    torch.manual_seed(42)
    original = make_pair(PAIR_MODE).relation_token
    torch.manual_seed(42)
    candidate = make_encoder(MODE).relation_token
    original_state = original.state_dict()
    candidate_state = candidate.state_dict()
    for name, value in original_state.items():
        assert name in candidate_state
        assert torch.equal(value, candidate_state[name]), name

    hidden = torch.linspace(-1.0, 1.0, steps=5 * 192).reshape(5, 192)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    membership = torch.zeros((5, 12))
    membership[0, 0] = 1
    membership[1, 3] = 1
    membership[3, 6] = 1
    _, original_diagnostics = original.compute_update(hidden, batch)
    update, candidate_diagnostics = candidate.compute_update(
        hidden, batch, membership
    )
    assert torch.equal(
        original_diagnostics["assignment"],
        candidate_diagnostics["assignment"],
    )
    assert torch.count_nonzero(candidate_diagnostics["role_bias"]).item() == 0
    assert torch.count_nonzero(update).item() == 0
