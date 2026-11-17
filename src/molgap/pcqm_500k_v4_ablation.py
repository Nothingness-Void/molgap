"""Causal global-communication ablations for the matched PCQM 500K V4 run."""
from __future__ import annotations


ABLATION_PARAMETERS = {
    "edge_local_only": 3_433_601,
    "edge_sparse_global_369": 3_879_425,
}
GLOBAL_LAYERS = (3, 6, 9)


def make_ablation_encoder(arm: str):
    """Build one frozen EdgeState ablation from the accepted shared backbone."""
    if arm not in ABLATION_PARAMETERS:
        raise ValueError(f"Unknown matched-500K ablation arm: {arm}")

    import torch.nn as nn

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper
    from .qm9_neural_atom import _LocalGPSBlockFactory

    model = OGBEdgeStateStructuralGPSWrapper(
        in_channels=9,
        edge_dim=3,
        hidden_channels=192,
        num_layers=9,
        num_heads=4,
        dropout=0.05,
        n_targets=1,
        pooling="mean",
        rwse_dim=16,
        edge_state_channels=64,
    )
    retained = GLOBAL_LAYERS if arm == "edge_sparse_global_369" else ()
    model.convs = nn.ModuleList(
        block
        if layer in retained
        else _LocalGPSBlockFactory.make(block)
        for layer, block in enumerate(model.convs, start=1)
    )
    model.v4_global_layers = tuple(retained)
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != ABLATION_PARAMETERS[arm]:
        raise RuntimeError(
            f"Ablation parameter identity changed: {arm}={observed}"
        )
    return model
