import json

import pytest
import torch

from molgap.pcqm_graph_state_continuation import (
    GraphStateContinuationConfig,
    _safe_artifact,
    convergence_update,
)


def test_convergence_requires_material_improvement() -> None:
    anchor, wait = convergence_update(0.1200, 3, 0.11996, 5e-5)
    assert anchor == 0.1200
    assert wait == 4
    anchor, wait = convergence_update(anchor, wait, 0.11990, 5e-5)
    assert anchor == pytest.approx(0.11990)
    assert wait == 0


def test_continuation_contract_is_bounded_and_fp32() -> None:
    config = GraphStateContinuationConfig()
    assert config.source_epochs == 12
    assert config.max_total_epochs == 80
    assert config.convergence_patience == 8
    assert config.segment_seconds < 4 * 3600
    assert config.precision == "fp32_tf32_off"


def test_source_artifact_path_cannot_escape(tmp_path) -> None:
    assert _safe_artifact(tmp_path, "validation/manifest.json") == tmp_path / "validation/manifest.json"
    with pytest.raises(RuntimeError, match="Unsafe"):
        _safe_artifact(tmp_path, "../outside.pt")
    with pytest.raises(RuntimeError, match="Unsafe"):
        _safe_artifact(tmp_path, str(tmp_path / "absolute.pt"))
