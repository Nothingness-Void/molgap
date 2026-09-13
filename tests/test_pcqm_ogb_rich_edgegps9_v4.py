import json
from pathlib import Path

import torch

from molgap.pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH,
    EPOCHS,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    TRAIN_ROWS,
    DeterministicEpochBatchSampler,
)
from molgap.pcqm_ogb_rich_gps_v4 import (
    MODEL_ID,
    MODEL_PARAMETERS,
    _validate_contract_file,
    _model,
    _scientific_fields,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_ogb_rich_edgegps9_100k_v4"


def test_v4_contract_matches_frozen_reference_screen_fields():
    contract = json.loads((EXPERIMENT / "training_contract.json").read_text())
    fields = _scientific_fields()
    for key, value in fields.items():
        assert contract[key] == value
    assert contract["model_id"] == MODEL_ID
    assert contract["parameter_count"] == MODEL_PARAMETERS
    assert contract["physical_batch_per_device"] == 128
    assert contract["total_optimizer_steps"] == BATCHES_PER_EPOCH * EPOCHS
    assert contract["sample_exposure"] == SAMPLE_PRESENTATIONS
    assert contract["role_access"]["official_validation_role_read"] is False
    assert contract["role_access"]["test_dev_role_read"] is False
    assert contract["role_access"]["test_challenge_role_read"] is False
    assert len(_validate_contract_file()) == 64


def test_model_identity_and_backward():
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS
    x = torch.zeros((6, 9), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long
    )
    edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    rwse = torch.zeros((6, 16), dtype=torch.float32)
    prediction = model(x, edge_index, edge_attr, batch, rwse).view(-1)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    prediction.abs().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_fixed_sampler_has_only_physical_bs128_batches():
    batches = list(DeterministicEpochBatchSampler(TRAIN_ROWS, 0))
    assert len(batches) == BATCHES_PER_EPOCH == 781
    assert all(len(batch) == PHYSICAL_BATCH for batch in batches)
    flat = [index for item in batches for index in item]
    assert len(flat) == len(set(flat)) == 99_968
