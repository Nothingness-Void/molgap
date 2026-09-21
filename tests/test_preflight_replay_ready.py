"""Tests for preflight replay-ready verification tool."""
from __future__ import annotations

from pathlib import Path
import pytest

from molgap.constants import REPO_ROOT
from molgap.preflight import (
    PreflightError,
    check_code_and_package_integrity,
    check_prospective_trajectory,
    run_triple_check,
)


def test_preflight_passes_for_prospective_500k():
    exp_dir = REPO_ROOT / "experiments" / "pcqm_gptrans_noisy_pair_norm_500k"
    pkg_dir = REPO_ROOT / "platforms" / "_records" / "kaggle" / "staging" / "pcqm_gptrans_noisy_pair_norm_500k"

    traj = check_prospective_trajectory(exp_dir)
    assert traj["trajectory_id"] == "TB-gptrans-noisy-pair-norm-500k-s42"
    assert traj["record_mode"] == "prospective"
    assert traj["decision"]["outcome"] == "ACTIVE"

    if pkg_dir.is_dir():
        meta = check_code_and_package_integrity(pkg_dir)
        assert "nvoid912" in meta["id"]

    run_triple_check(exp_dir, pkg_dir if pkg_dir.is_dir() else None)


def test_preflight_fails_on_missing_or_terminal_trajectory(tmp_path: Path):
    empty_dir = tmp_path / "bad_exp"
    empty_dir.mkdir()
    with pytest.raises(PreflightError, match="Missing prospective trajectory"):
        check_prospective_trajectory(empty_dir)

    traj_path = empty_dir / "trajectory.json"
    traj_path.write_text('{"schema": "molgap-trajectory-v1", "record_mode": "retrospective_partial", "decision": {"outcome": "ACTIVE"}}', encoding="utf-8")
    with pytest.raises(PreflightError, match="record_mode must be 'prospective'"):
        check_prospective_trajectory(empty_dir)
