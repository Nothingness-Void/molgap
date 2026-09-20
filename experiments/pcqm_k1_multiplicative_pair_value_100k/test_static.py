from __future__ import annotations

import json
from pathlib import Path

from molgap.k1_multiplicative_pair_value import (
    ADDED_PARAMETERS,
    MODE,
    PARAMETERS,
    make_encoder,
)
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS


ROOT = Path(__file__).resolve().parent


def test_frozen_contract_identity():
    contract = json.loads((ROOT / "training_contract.json").read_text())
    assert contract["arm"] == MODE
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["epochs"] == 40
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["total_sample_presentations"] == 3_998_720


def test_architecture_registry_is_value_only():
    config = ARCHITECTURE_CONFIGS[MODE]
    assert config["pair_selection"] == "accepted-additive-learned-query"
    assert config["pair_value"] == "separate-ordered-low-rank-hadamard-product"
    assert config["added_parameters"] == ADDED_PARAMETERS
    assert config["expected_parameters"] == PARAMETERS[MODE]
    assert config["geometry"] is False
    assert config["external_features"] is False


def test_model_parameter_identity_and_zero_nested_initialization():
    import torch

    torch.manual_seed(42)
    model = make_encoder(MODE)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[MODE]
    assert torch.count_nonzero(model.relation_token.return_projection.weight) == 0
    assert model.relation_token.selection_source is not model.relation_token.value_source
    assert not (
        set(model.relation_token.selection_source.parameters())
        & set(model.relation_token.value_source.parameters())
    )


def test_kaggle3_single_t4_binding():
    metadata = json.loads(
        (ROOT / "kaggle_candidate/kernel-metadata.json").read_text()
    )
    assert metadata["id"] == "nvoid912/molgap-k1-multiplicative-pairvalue-s42"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "nvoid912/molgap-k1-multiplicative-pairvalue-source",
        "nvoid912/pcqm4mv2-ogb-fixed-100k-v1",
    ]
