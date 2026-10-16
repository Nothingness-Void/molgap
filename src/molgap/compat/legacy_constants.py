"""Historical constants retained for old imports and closed-study readers.

Registry-backed paths and inference defaults remain in ``molgap.constants``.
Importing their roots here preserves accepted asset locations and avoids a
second path policy; the constants module imports this owner only on access.
"""
from __future__ import annotations

from ..constants import (
    MODELS_DIR as _MODELS_DIR,
    PARAMS_SCHNET_300K as _PARAMS_SCHNET_300K,
    PRODUCTION_HISTORY_DIR as _PRODUCTION_HISTORY_DIR,
    RAW_DIR as _RAW_DIR,
)

DATA_PHASE3 = _RAW_DIR / "phase3_chonsfcl_mw200_500_30k.csv"
DATA_PHASE6_LARGE = _RAW_DIR / "phase6_chonsfcl_mw500_1000_15k.csv"
GRAPHS_PHASE4 = _PRODUCTION_HISTORY_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
MODEL_PHASE4 = _MODELS_DIR / "gnn_schnet_3d_tuned.pt"

PARAMS_PHASE4 = {
    "hidden_channels": 192,
    "num_filters": 256,
    "num_interactions": 6,
    "num_gaussians": 100,
    "cutoff": 6.0,
    "dropout": 0.2,
}

# The A/B baseline was a copy, not an alias of the inference parameters.
PARAMS_AB_SCHNET = dict(_PARAMS_SCHNET_300K)

# Closed ab3d arms used smaller equivariant models for the 8 GB RTX 5060.
PARAMS_VISNET = {
    "hidden_channels": 128,
    "num_layers": 4,
    "num_heads": 8,
    "num_rbf": 32,
    "cutoff": 5.0,
    "dropout": 0.0,
}

PARAMS_TENSORNET = {
    "hidden_channels": 128,
    "num_layers": 2,
    "num_rbf": 32,
    "cutoff": 5.0,
    "dropout": 0.0,
}

AB_ENCODERS = {
    "schnet": {"kind": "schnet", "params": PARAMS_AB_SCHNET, "use_charges": True},
    "visnet": {"kind": "visnet", "params": PARAMS_VISNET, "use_charges": False},
    "tensornet": {"kind": "tensornet", "params": PARAMS_TENSORNET, "use_charges": False},
}
