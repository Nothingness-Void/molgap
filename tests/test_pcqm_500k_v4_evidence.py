import math
import importlib.util
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_500k_v4_ablation import (
    ABLATION_PARAMETERS,
    GLOBAL_LAYERS,
    make_ablation_encoder,
)
from molgap.pcqm_500k_v4_evidence import scientific_contract, schedule


def _analysis_module():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "pcqm_500k_v4_evidence"
        / "analyze_local_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("pcqm_v4_ablation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exposure_excludes_every_tail_batch():
    contract = scientific_contract()
    assert contract['sample_exposure'] == 29998080
    assert contract['steps_per_epoch'] == 3906
    assert contract['sample_exposure'] % 128 == 0
    assert 500000 - contract['steps_per_epoch'] * 128 == 32


def test_joint_regularization_arm_has_truthful_frozen_contract():
    from molgap.pcqm_500k_v4_evidence import PARAMETERS

    contract = scientific_contract("gptrans_noisy_pair_norm")
    assert PARAMETERS["gptrans_noisy_pair_norm"] == 5_277_400
    assert contract["loss_fingerprint"] == (
        "normalized-gap-l1-plus-noisy-nodes-ce-alpha0.1"
    )
    assert contract["sample_exposure"] == 29_998_080


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


def test_weighted_median_blend_finds_exact_l1_solution():
    analysis = _analysis_module()
    target = np.array([0.25, 0.75, 1.25, 1.75])
    first = np.array([1.0, 1.0, 2.0, 2.0])
    second = np.array([0.0, 0.0, 1.0, 1.0])
    weight = analysis._weighted_median_weight(first, second, target)
    assert weight == 0.25


def test_crossfit_pair_keeps_every_row_held_out_once():
    analysis = _analysis_module()
    source_idx = np.arange(20)
    target = source_idx.astype(np.float64) / 10
    first = target + 0.1
    second = target - 0.1
    result = analysis._crossfit_pair(first, second, target, source_idx)
    assert len(result["fold_weights_first"]) == 5
    assert math.isclose(result["mean_weight_first"], 0.5)
    assert result["oof_mae_eV"] < 1e-12
