"""Test-time import helpers for CLI modules that live outside `src/`.

Reusable logic lives in `src/molgap/`. A few tests still cover helper functions
that belong to a thin CLI (argument parsing, split manifests), and those CLIs sit
under `production/` or `experiments/` rather than in the installed package. This
maps a stable dotted name onto each such file so the tests do not encode the
directory layout, which is the thing most likely to move again.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# dotted alias -> path relative to the repository root
CLI_MODULES = {
    "cli.train_encoder": "production/03_train/scripts/training/train_encoder.py",
    "cli.eval_multi2d_experts": (
        "production/04_evaluate/scripts/evaluation/eval_multi2d_experts.py"
    ),
    "cli.build_repaired_2m_schnet_ab_subset": (
        "production/02_graphs/scripts/data/build_repaired_2m_schnet_ab_subset.py"
    ),
    "cli.train_three_gps_embedding_residual": (
        "experiments/_scripts/train_three_gps_embedding_residual.py"
    ),
    "cli.train_hierarchical_2d3d_fusion": (
        "experiments/_scripts/train_hierarchical_2d3d_fusion.py"
    ),
}


def _load(alias: str, relative: str) -> None:
    path = REPO_ROOT / relative
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


for _alias, _relative in CLI_MODULES.items():
    _load(_alias, _relative)
