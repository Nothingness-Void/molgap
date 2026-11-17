"""Historical exports with unchanged desktop asset paths and parameter values."""
from __future__ import annotations

from ..constants import (
    LEGACY_RAW_DIR as _LEGACY_RAW_DIR,
    LEGACY_CACHE_DIR as _LEGACY_CACHE_DIR,
    LEGACY_MODEL_DIR as _LEGACY_MODEL_DIR,
    PARAMS_SCHNET_300K as _PARAMS_SCHNET_300K,
)

DATA_PHASE3 = _LEGACY_RAW_DIR / "phase3_chonsfcl_mw200_500_30k.csv"

DATA_PHASE6_LARGE = _LEGACY_RAW_DIR / "phase6_chonsfcl_mw500_1000_15k.csv"

GRAPHS_PHASE4 = _LEGACY_CACHE_DIR / "pyg_3d_graphs_etkdg.pt"

MODEL_PHASE4 = _LEGACY_MODEL_DIR / "gnn_schnet_3d_tuned.pt"

PARAMS_PHASE4 = {
    "hidden_channels": 192,
    "num_filters": 256,
    "num_interactions": 6,
    "num_gaussians": 100,
    "cutoff": 6.0,
    "dropout": 0.2,
}

PARAMS_AB_SCHNET = dict(_PARAMS_SCHNET_300K)  # invariant baseline, deployed form

PARAMS_VISNET = {
    "hidden_channels": 128,
    "num_layers": 4,
    "num_heads": 8,      # 128 % 8 == 0
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
    "schnet":    {"kind": "schnet",    "params": PARAMS_AB_SCHNET,  "use_charges": True},
    "visnet":    {"kind": "visnet",    "params": PARAMS_VISNET,     "use_charges": False},
    "tensornet": {"kind": "tensornet", "params": PARAMS_TENSORNET,  "use_charges": False},
}
