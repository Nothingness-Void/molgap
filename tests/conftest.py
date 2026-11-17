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
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# dotted alias -> path relative to the repository root
CLI_MODULES = {
    "cli.build_repaired_2m_schnet_ab_subset": (
        "production/02_graphs/scripts/data/build_repaired_2m_schnet_ab_subset.py"
    ),
}


def _load(alias: str, relative: str):
    if alias in sys.modules:
        return sys.modules[alias]
    path = REPO_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"CLI alias {alias!r} points to missing file: {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for CLI alias {alias!r}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


@pytest.fixture
def cli_module():
    """Only the test asking for a CLI imports it; normal collection stays light."""
    def load(alias):
        return _load(alias, CLI_MODULES[alias])
    return load
