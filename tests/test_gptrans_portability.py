"""Synthetic retained tensors/metadata only; no model, cache or GPU execution."""
import ast
import json
from pathlib import Path

import pytest
import torch

from molgap.gptrans_portability import (
    check_rows, check_reproduction, check_barrier, load_progress,
)


def predictions(start=100000, count=50000):
    return dict(source_idx=torch.arange(start, start+count), target_eV=torch.ones(count),
                prediction_eV=torch.full((count,), 1.1))


def test_original_reproduction():
    assert check_reproduction(predictions(), predictions())["accepted"]


@pytest.mark.parametrize("field", ["source_idx", "target_eV", "prediction_eV"])
def test_reproduction_mismatch(field):
    actual = predictions()
    actual[field][5] += 1
    with pytest.raises(ValueError):
        check_reproduction(actual, predictions())


def test_nonfinite_rejected():
    row = predictions(count=3)
    row["prediction_eV"][0] = float("nan")
    with pytest.raises(ValueError):
        check_rows(row, 100000, 3)


def test_reproduction_mae_tolerance():
    actual = predictions()
    actual["prediction_eV"] += 0.0005
    with pytest.raises(ValueError):
        check_reproduction(actual, predictions())


def test_role_and_partial_batch():
    check_rows(predictions(500000, 80), 500000, 80)
    with pytest.raises(ValueError):
        check_rows(predictions(500000, 80), 100000, 80)


def test_both_reproductions_required(tmp_path):
    identity = {"source_commit": "frozen"}
    assert not check_barrier(tmp_path, identity)
    for arm in ("ema999", "ema9999"):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "reproduction.json").write_text(json.dumps(dict(identity=identity, accepted=True)))
        assert check_barrier(tmp_path, identity) is (arm == "ema9999")
    with pytest.raises(ValueError):
        check_barrier(tmp_path, {"source_commit": "different"})


def test_unrecorded_chunks_fail_closed(tmp_path):
    (tmp_path / "chunk_00.pt").write_bytes(b"not accepted")
    with pytest.raises(ValueError):
        load_progress(tmp_path, {})


def test_resume_identity_fail_closed(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(dict(identity={"model": "old"}, chunks=[])))
    with pytest.raises(ValueError):
        load_progress(tmp_path, {"model": "new"})


def test_no_optimizer_or_reapplication():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/molgap/gptrans_portability.py").read_text()
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any("optim." in name or "apply_author_variant" in name or "EMA" in name for name in calls)
    assert "drop_last=False" in source
    assert "configure_fp32_determinism(42)" in source
    assert source.index("while not check_barrier") < source.index('graphs = graphs_for("unseen_500k")')
