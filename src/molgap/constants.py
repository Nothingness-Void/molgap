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

SEED = 42
