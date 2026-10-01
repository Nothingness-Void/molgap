"""Fixed K1 node-width addon; edge/slot widths and depth retain family recipe."""
from __future__ import annotations

CONFIG = {"hidden_channels": 256, "num_layers": 9,
          "edge_state_channels": 64, "latent_channels": 64, "active_slots": 1}
EXPECTED_PARAMETER_COUNT = 6_035_201
REFERENCE_PARAMETER_COUNT = 3_658_817


def make_encoder():
    """Construct on CPU; the caller owns seed and frozen-state publication."""
    from .qm9_neural_atom import make_encoder as make_family_encoder
    model = make_family_encoder("neural_atom_k1", hidden_channels=256)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("K1 width256 parameter identity changed")
    return model
