"""No-model-execution checks for the prospective precision comparison."""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from molgap.pcqm_k1_tf32_comparison import (
    ARMS,
    BATCH_SIZE,
    EPOCHS,
    MODEL_MODE,
    PARAMETERS,
    SAMPLE_EXPOSURE,
    STEPS_PER_EPOCH,
)


def test_frozen_contract_constants() -> None:
    assert ARMS == ("fp32", "tf32_matmul")
    assert MODEL_MODE == "neural_atom_k1_v4"
    assert PARAMETERS == 3_658_817
    assert BATCH_SIZE == 128
    assert EPOCHS == 40
    assert STEPS_PER_EPOCH == 781
    assert SAMPLE_EXPOSURE == 3_998_720


def test_pbs_rejects_unqualified_cache_root() -> None:
    script = Path(__file__).with_name("train_a100.pbs").read_text()
    assert 'realpath "$MOLGAP_TF32_CACHE_ROOT"' in script
    assert 'case "$MOLGAP_TF32_CACHE_ROOT"' in script
    assert "nvidia-smi -L" in script
    assert "walltime=10:00:00" in script


def test_protocol_preserves_protected_roles() -> None:
    protocol = Path(__file__).with_name("protocol.md").read_text()
    assert "official validation, test-dev, test-challenge" in protocol
    assert "No model/module/data changes" in protocol


def test_incomplete_comparison_fails_closed(tmp_path: Path) -> None:
    from experiments.pcqm_k1_tf32_runtime_comparison_100k.accept import accept

    (tmp_path / "comparison.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(RuntimeError, match="incomplete"):
        accept(tmp_path)


def test_canonical_trace_records_observed_checkpoint(tmp_path: Path) -> None:
    from molgap.pcqm_k1_tf32_comparison import _canonical_recorder, _record_epoch, _write_role_history
    from molgap.research_memory.trace import load_canonical_trace

    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"test-only checkpoint identity")
    row = {
        "epoch": 0,
        "optimizer_steps": STEPS_PER_EPOCH,
        "sample_presentations": BATCH_SIZE * STEPS_PER_EPOCH,
        "train_normalized_mae": 0.2,
        "development_gap_mae_eV": 0.3,
        "learning_rate": 4e-4,
        "training_seconds": 1.0,
        "validation_seconds": 0.5,
    }
    recorder = _canonical_recorder(tmp_path / "canonical_trace.json", "fp32")
    _record_epoch(recorder, row, checkpoint, [row])
    _write_role_history(tmp_path, "fp32", 0)
    trace = load_canonical_trace(tmp_path / "canonical_trace.json")
    assert trace["observations"][0]["checkpoint_identity"]
    assert trace["observations"][0]["sample_presentations"] == 99_968
    assert json.loads((tmp_path / "role_history.json").read_text())["last_observed_epoch"] == 0
