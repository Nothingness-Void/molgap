"""OGB EdgeState architecture variants; no training or submission authority."""
from __future__ import annotations


from .experiment_spec import (
    EDGE_STATE_BASE_LAYERS as BASE_LAYERS,
    EDGE_STATE_MAX_LAYERS as MAX_LAYERS,
    EDGE_STATE_MIN_LAYERS as MIN_LAYERS,
)


def make_encoder(num_layers: int = BASE_LAYERS):
    """Build the frozen nine-layer backbone or its depth-only counterpart."""
    if type(num_layers) is not int or not MIN_LAYERS <= num_layers <= MAX_LAYERS:
        raise ValueError(f"num_layers must be an integer in [{MIN_LAYERS}, {MAX_LAYERS}]")
    if num_layers == BASE_LAYERS:
        from .qm9_local_hierarchy import make_encoder as make_base

        return make_base()

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    return OGBEdgeStateStructuralGPSWrapper(
        in_channels=9,
        edge_dim=3,
        hidden_channels=192,
        num_layers=num_layers,
        num_heads=4,
        dropout=0.05,
        n_targets=1,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
    )
