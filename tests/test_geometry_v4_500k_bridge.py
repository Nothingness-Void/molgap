"""Static release checks; GPU forward/backward stays in remote preflight."""
import json
from pathlib import Path

import torch

from molgap.gptrans import OGBGPTransTiny
from molgap.pcqm_500k_v4_evidence import (
    BS, EPOCHS, GEOMETRY_DEVICE_SECOND_CAPS, PARAMETERS, STEPS,
    make_model, schedule, scientific_contract,
)
from molgap.pcqm_geometry_transfer import GeometryGPTransTiny


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((
    ROOT / "experiments/pcqm_geometry_v4_500k_pair/training_contract.json"
).read_text(encoding="utf-8"))


def test_geometry_bridge_freezes_the_matched_v4_recipe():
    assert CONTRACT["arms"] == ["gptrans_distance_only", "k1_distance_angle"]
    assert CONTRACT["epochs"] == EPOCHS == 60
    assert CONTRACT["physical_batch_per_device"] == BS == 128
    assert CONTRACT["steps_per_epoch"] == STEPS == 3906
    assert CONTRACT["optimizer_steps_per_arm"] == EPOCHS * STEPS
    assert CONTRACT["sample_presentations_per_arm"] == EPOCHS * STEPS * BS
    assert schedule(0) == CONTRACT["learning_rate_start"]
    assert schedule(EPOCHS - 1) == CONTRACT["learning_rate_end"]
    for arm in CONTRACT["arms"]:
        candidate = scientific_contract(arm)
        reference = scientific_contract(CONTRACT["references"][arm]["arm"])
        changed = {key for key in candidate if candidate[key] != reference[key]}
        assert changed == {"feature_fingerprint"}
        assert candidate["sample_exposure"] == CONTRACT["sample_presentations_per_arm"]
        assert GEOMETRY_DEVICE_SECOND_CAPS[arm] == CONTRACT["total_device_hour_cap"][arm] * 3600


def test_geometry_models_are_zero_start_and_keep_gptrans_core_initialization():
    torch.manual_seed(42)
    reference = OGBGPTransTiny(
        node_channels=256, pair_channels=32, num_layers=12, num_heads=8,
        shortest_path_cap=20, dropout=0.1, drop_path=0.1,
        layer_scale=1.0, n_targets=1,
    )
    torch.manual_seed(42)
    candidate = GeometryGPTransTiny(geometry_mode="distance_only")
    assert candidate.geometry_mode == "distance_only"
    assert sum(p.numel() for p in candidate.parameters()) == PARAMETERS["gptrans_distance_only"]
    assert all(torch.equal(value, candidate.state_dict()[key])
               for key, value in reference.state_dict().items())
    assert torch.count_nonzero(candidate.distance_to_pair.weight) == 0
    assert torch.count_nonzero(candidate.angle_to_node.weight) == 0

    torch.manual_seed(42)
    k1 = make_model("k1_distance_angle")
    assert sum(p.numel() for p in k1.parameters()) == PARAMETERS["k1_distance_angle"]
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in k1.wedge_to_edge)
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in k1.wedge_to_node)
