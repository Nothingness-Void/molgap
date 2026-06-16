"""MolGap — OLED molecular property prediction.

Re-exports the cheap path/config constants eagerly. The inference API
(``predict_smiles``, ``load_hybrid``, …) is exposed lazily via ``__getattr__``
so ``import molgap`` does not pull in torch / torch_geometric until you
actually run a prediction.
"""
from .constants import (
    REPO_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    TARGET_COLS, METADATA_COLS,
    DATA_PHASE3, DATA_PHASE6_LARGE,
    GRAPHS_PHASE4, GRAPHS_PHASE6,
    MODEL_PHASE4, MODEL_PHASE6,
    PARAMS_PHASE4, PARAMS_PHASE6,
    MODEL_REGISTRY, SEED,
)

# Lazily forwarded to molgap.inference on first access (keeps torch out of import).
_LAZY_API = {
    "predict_smiles",
    "predict_smiles_batch",
    "predict_smiles_batch_hybrid",
    "predict_smiles_ensemble",
    "load_model",
    "load_hybrid",
}


def __getattr__(name):
    if name in _LAZY_API:
        from . import inference
        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_API])
