from __future__ import annotations

import torch

from molgap.gps import GPSWrapper
from molgap.pcqm_route_b_training import CONFIGS, expand_gps_input_state


def test_route_b_configs_preserve_frozen_architectures() -> None:
    assert CONFIGS["gps9"].hidden_channels == 192
    assert CONFIGS["gps9"].num_layers == 9
    assert CONFIGS["gps11_160"].hidden_channels == 160
    assert CONFIGS["gps11_160"].num_layers == 11
    assert CONFIGS["primary_schnet"].augmented is False
    assert CONFIGS["augmented_schnet"].augmented is True


def test_expand_gps_input_state_preserves_old_feature_mapping() -> None:
    old = GPSWrapper(
        in_channels=9,
        hidden_channels=16,
        num_layers=1,
        num_heads=4,
        n_targets=3,
    )
    new = GPSWrapper(
        in_channels=18,
        hidden_channels=16,
        num_layers=1,
        num_heads=4,
        n_targets=3,
    )
    with torch.no_grad():
        old.node_emb.weight.copy_(
            torch.arange(16 * 9, dtype=torch.float32).view(16, 9)
        )
    state, report = expand_gps_input_state(
        old.state_dict(), new.state_dict()
    )
    assert torch.equal(
        state["node_emb.weight"][:, :6], old.node_emb.weight[:, :6]
    )
    assert torch.equal(
        state["node_emb.weight"][:, 15:18], old.node_emb.weight[:, 6:9]
    )
    assert torch.count_nonzero(state["node_emb.weight"][:, 6:15]) == 0
    assert report["target_input_dim"] == 18
