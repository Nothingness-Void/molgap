"""Centralized path constants and model configurations."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
COMMERCIAL_DIR = DATA_DIR / "commercial"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SCRIPTS_DIR = REPO_ROOT / "scripts"

TARGET_COLS = ["homo", "lumo", "gap"]
METADATA_COLS = ["cid", "mw", "formula", "smiles", "canonical_smiles"]

# ── Data files ──

DATA_PHASE3 = RAW_DIR / "phase3_chonsfcl_mw200_500_30k.csv"
DATA_PHASE6_LARGE = RAW_DIR / "phase6_chonsfcl_mw500_1000_15k.csv"

# ── Graph caches ──

GRAPHS_PHASE4 = RESULTS_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
GRAPHS_PHASE6 = RESULTS_DIR / "phase6" / "pyg_3d_graphs_etkdg_expanded.pt"

# ── Model checkpoints ──

MODEL_PHASE4 = MODELS_DIR / "gnn_schnet_3d_tuned.pt"
MODEL_PHASE6 = MODELS_DIR / "gnn_schnet_3d_optuna_expanded.pt"

# Phase 7 (300k, raw eV — no normalization)
MODEL_SCHNET_300K = MODELS_DIR / "gnn_schnet_3d_300k.pt"
MODEL_GPS_2D = MODELS_DIR / "gps_2d_300k.pt"
MODEL_HYBRID = MODELS_DIR / "hybrid_fusion_optuna.pt"
FUSION_METRICS = RESULTS_DIR / "phase7" / "fusion_optuna_metrics.json"

# Phase 8 replacement300k v2 candidate (raw eV — no normalization)
MODEL_PHASE8_REPLACEMENT_GPS = MODELS_DIR / "phase8_gps_replacement_300k.pt"
MODEL_PHASE8_REPLACEMENT_SCHNET = MODELS_DIR / "phase8_schnet_replacement_300k.pt"
MODEL_PHASE8_REPLACEMENT_HYBRID = MODELS_DIR / "phase8_hybrid_fusion_replacement_300k.pt"
FUSION_PHASE8_REPLACEMENT_METRICS = RESULTS_DIR / "phase8" / "fusion_replacement_300k_metrics.json"

# TensorNet — ab3d experimental 3D encoder (NOT production). Solo TensorNet beats
# SchNet, but at fusion level the gap collapses to <0.2% R² while costing ~3.7x
# training time at 1M scale, so production stays on SchNet. See CURRENT_STATE.md
# and results/ab3d/comparison.md. These artifacts are kept for the A/B record.
MODEL_TENSORNET_300K = MODELS_DIR / "tensornet_3d_300k.pt"
MODEL_HYBRID_TENSORNET = MODELS_DIR / "hybrid_fusion_tensornet.pt"
FUSION_TENSORNET_METRICS = RESULTS_DIR / "phase7" / "fusion_tensornet_metrics.json"

# ── Model hyperparameters ──

PARAMS_PHASE4 = {
    "hidden_channels": 192,
    "num_filters": 256,
    "num_interactions": 6,
    "num_gaussians": 100,
    "cutoff": 6.0,
    "dropout": 0.2,
}

PARAMS_PHASE6 = {
    "hidden_channels": 192,
    "num_filters": 256,
    "num_interactions": 6,
    "num_gaussians": 100,
    "cutoff": 8.0,
    "dropout": 0.1,
}

PARAMS_SCHNET_300K = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}

PARAMS_GPS_2D = {
    "hidden_channels": 192,
    "num_layers": 7,
    "num_heads": 4,
    "dropout": 0.05,
}

# ── A/B 3D-encoder comparison (scripts/ab3d) ──
# Same hidden=192 across encoders for capacity parity; layer counts follow each
# architecture's convention. Param counts are reported by train_encoder.py.
PARAMS_AB_SCHNET = dict(PARAMS_SCHNET_300K)  # invariant baseline, deployed form

# Feasibility-tuned for the RTX 5060 (8 GB, 30 SM): the equivariant/tensor nets
# are 40-130x heavier per batch than SchNet at hidden=192, so they run at
# hidden=128 with fewer layers + cutoff 5.0 (fewer edges). Capacity therefore
# differs from SchNet (h192) — param counts are reported and the gap is noted in
# the comparison; the speed gap is itself a decision-relevant deliverable.
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

# Experimental TensorNet for 300k training (same arch as ab3d winner; not production)
PARAMS_TENSORNET_300K = {
    "hidden_channels": 128,
    "num_layers": 2,
    "num_rbf": 32,
    "cutoff": 5.0,
    "dropout": 0.0,
}

# Single source of truth for the A/B arms. `kind` selects the wrapper class in
# scripts/ab3d/train_encoder.py; `use_charges` is each encoder's native form
# (SchNet uses Gasteiger charges = deployed form; equivariant nets use Z+geometry).
AB_ENCODERS = {
    "schnet":    {"kind": "schnet",    "params": PARAMS_AB_SCHNET,  "use_charges": True},
    "visnet":    {"kind": "visnet",    "params": PARAMS_VISNET,     "use_charges": False},
    "tensornet": {"kind": "tensornet", "params": PARAMS_TENSORNET,  "use_charges": False},
}

# ── Model registry ──
# Single source of truth for "which checkpoint + which hyperparams + is it
# normalized". Consumed by inference.load_model(key=...) and inference.load_hybrid().
# kind: "schnet" → SchNetWrapper, "gps" → GPSWrapper, "hybrid" → FusionHead trio.
# normalized: True → predictions are (raw * y_std + y_mean); False → raw eV.
MODEL_REGISTRY = {
    "phase6_schnet": {
        "kind": "schnet", "checkpoint": MODEL_PHASE6, "params": PARAMS_PHASE6,
        "normalized": True, "graphs": GRAPHS_PHASE6, "use_charges": True,
    },
    "phase7_schnet_300k": {
        "kind": "schnet", "checkpoint": MODEL_SCHNET_300K, "params": PARAMS_SCHNET_300K,
        "normalized": False, "use_charges": True,
    },
    "phase7_gps_2d": {
        "kind": "gps", "checkpoint": MODEL_GPS_2D, "params": PARAMS_GPS_2D,
        "normalized": False,
    },
    "phase7_hybrid": {
        "kind": "hybrid", "checkpoint": MODEL_HYBRID, "metrics": FUSION_METRICS,
        "normalized": False, "components": ["phase7_gps_2d", "phase7_schnet_300k"],
    },
    "phase8_replacement_gps_2d": {
        "kind": "gps", "checkpoint": MODEL_PHASE8_REPLACEMENT_GPS, "params": PARAMS_GPS_2D,
        "normalized": False,
    },
    "phase8_replacement_schnet_300k": {
        "kind": "schnet", "checkpoint": MODEL_PHASE8_REPLACEMENT_SCHNET,
        "params": PARAMS_SCHNET_300K, "normalized": False, "use_charges": True,
    },
    "phase8_replacement_hybrid": {
        "kind": "hybrid", "checkpoint": MODEL_PHASE8_REPLACEMENT_HYBRID,
        "metrics": FUSION_PHASE8_REPLACEMENT_METRICS, "normalized": False,
        "components": ["phase8_replacement_gps_2d", "phase8_replacement_schnet_300k"],
        "fusion_type": "gate", "hidden": 192, "dropout": 0.0,
    },
    "tensornet_300k": {
        "kind": "tensornet", "checkpoint": MODEL_TENSORNET_300K,
        "params": PARAMS_TENSORNET_300K, "normalized": False, "use_charges": False,
    },
    "hybrid_tensornet": {
        "kind": "hybrid", "checkpoint": MODEL_HYBRID_TENSORNET,
        "metrics": FUSION_TENSORNET_METRICS,
        "normalized": False, "components": ["phase7_gps_2d", "tensornet_300k"],
    },
}

SEED = 42
