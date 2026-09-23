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
