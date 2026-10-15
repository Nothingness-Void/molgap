"""Cross-owner scientific axes and complete fixed100K timing horizons."""
import pytest

from molgap.execution_analysis import paired_expectations, paired_spans
from molgap.execution_terminal import excluded_trace_reference


def test_complete_sixty_pass_horizon_keeps_last_gptrans_span():
    rows = [{"start_relative_update": p * 781, "optimizer_updates": 781,
        "sample_presentations": 99968, "wall_seconds": 100.} for p in range(1, 60)]
    faster = [{**row, "wall_seconds": 80.} for row in rows]
    result = paired_spans(rows, faster, total_passes=60)
    assert len(result["passes"]) == 59
    assert result["passes"][-1]["pass"] == 60
    assert result["aggregate"]["speedup_factor"] == 1.25
    with pytest.raises(ValueError):
        paired_spans(rows, faster)
    with pytest.raises(ValueError):
        paired_spans(rows, faster[:-1], total_passes=60)


def test_epoch_and_cursor_contracts_match_without_relabeling_inspector_inputs():
    common = {"optimizer_steps": 46860, "sample_presentations": 5998080,
        "development_rows": 50000, "precision": "fp32", "source_idx_sha256": "a" * 64,
        "target_sha256": "b" * 64}
    old = {**common, "epochs": 60}
    new = {**common, "completed_passes": 60, "next_batch": 0}
    assert paired_expectations(old, new)["passes"] == 60
    assert "epochs" not in new
    for changed in ({"completed_passes": 59}, {"next_batch": 1},
                    {"target_sha256": "c" * 64}, {"precision": "bf16"}):
        with pytest.raises(ValueError):
            paired_expectations(old, {**new, **changed})


def test_excluded_trace_keeps_frozen_prior_without_granting_replay():
    def plan(refs):
        return {"state_at_start": {"reference_ids": refs}}
    assert excluded_trace_reference(plan([]), "self")[:2] == ("self", "reference")
    assert excluded_trace_reference(plan(["frozen-prior"]), "self")[:2] == ("frozen-prior", "candidate")
    with pytest.raises(ValueError):
        excluded_trace_reference(plan(["a", "b"]), "self")
