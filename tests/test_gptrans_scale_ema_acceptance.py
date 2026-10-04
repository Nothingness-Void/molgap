"""Synthetic retention tests; no model, data or remote execution."""
from molgap.gptrans_scale_ema_acceptance import _retained_artifacts
from molgap.research_memory.terminal_wiring import inspect_trace_retention_evidence, is_trace_artifact
from molgap.training_reproducibility import atomic_json


def test_shared_live_views_retain_one_unambiguous_canonical_trace(tmp_path):
    paths = [tmp_path / view / name for view in ("ema999", "ema9999")
        for name in ("canonical_trace.json", "trace.json")]
    for path in paths:
        atomic_json(path, {"synthetic": True})
    primary = tmp_path / "ema999/canonical_trace.json"
    artifacts = _retained_artifacts(paths, primary, lambda p: p.relative_to(tmp_path).as_posix())
    assert len(artifacts) == 4
    assert sum(is_trace_artifact(a) for a in artifacts) == 1
    declared, selected = inspect_trace_retention_evidence({"artifacts": artifacts}, arm_identifier="ema999")
    assert declared and selected["locator"] == "ema999/canonical_trace.json"
    assert all(a["sha256"] and a["availability"] == "local_verified" for a in artifacts)
