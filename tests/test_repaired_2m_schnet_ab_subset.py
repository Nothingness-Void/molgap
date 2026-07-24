from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase8"
    / "data"
    / "build_repaired_2m_schnet_ab_subset.py"
)
SPEC = importlib.util.spec_from_file_location("build_repaired_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_proportional_sample_is_deterministic_and_unique() -> None:
    frame = pd.DataFrame(
        {
            "manifest_row": range(100),
            "source_group": ["base"] * 70 + ["repair"] * 30,
            "joint_bucket": ["a"] * 40 + ["b"] * 30 + ["a"] * 20 + ["b"] * 10,
        }
    )
    first = MODULE.proportional_sample(frame, 20, seed=42)
    second = MODULE.proportional_sample(frame, 20, seed=42)
    assert first.tolist() == second.tolist()
    assert len(first) == len(set(first.tolist())) == 20
