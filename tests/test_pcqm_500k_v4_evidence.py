import math
import torch

from molgap.pcqm_500k_v4_ablation import (
    ABLATION_PARAMETERS,
    GLOBAL_LAYERS,
    make_ablation_encoder,
)
from molgap.pcqm_500k_v4_evidence import scientific_contract, schedule


def test_exposure_excludes_every_tail_batch():
    contract = scientific_contract()
    assert contract['sample_exposure'] == 29998080
    assert contract['steps_per_epoch'] == 3906
    assert contract['sample_exposure'] % 128 == 0
    assert 500000 - contract['steps_per_epoch'] * 128 == 32


def test_schedule_survives_stage_boundaries():
    whole = [schedule(epoch) for epoch in range(60)]
    staged = [schedule(epoch) for start in range(0, 60, 4) for epoch in range(start, start+4)]
    assert whole == staged
    assert math.isclose(whole[0], 4e-4)
    assert math.isclose(whole[-1], 1e-6)
    assert all(a >= b for a, b in zip(whole, whole[1:]))


def test_ablation_parameter_and_global_layer_identities():
    for arm, expected in ABLATION_PARAMETERS.items():
        model = make_ablation_encoder(arm)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected
        observed = tuple(
            layer
            for layer, block in enumerate(model.convs, start=1)
            if hasattr(block, "attn")
        )
        expected_layers = GLOBAL_LAYERS if arm == "edge_sparse_global_369" else ()
        assert observed == expected_layers


def test_ablation_forward_shapes():
    x = torch.zeros((6, 9), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    rwse = torch.zeros((6, 16), dtype=torch.float32)
    for arm in ABLATION_PARAMETERS:
        output = make_ablation_encoder(arm)(x, edge_index, edge_attr, batch, rwse)
        assert output.shape == (2, 1)
        assert torch.isfinite(output).all()
