from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_train_encoder():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "phase8"
        / "training"
        / "train_encoder.py"
    )
    spec = importlib.util.spec_from_file_location("phase8_train_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_roles_are_seeded_and_balanced() -> None:
    module = _load_train_encoder()
    first = module._split_roles(100, 42)
    repeated = module._split_roles(100, 42)
    alternate = module._split_roles(100, 43)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, alternate)
    assert [(first == role).sum() for role in range(3)] == [80, 10, 10]
