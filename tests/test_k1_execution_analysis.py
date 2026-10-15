"""Saved-artifact arithmetic, alignment and overlapping GPU trace boundaries."""
import json
import pytest
import torch

from molgap.k1_execution_analysis import paired_spans, prediction_comparison, speed, trace_observations


def test_speed_keeps_time_reduction_distinct_from_throughput_gain():
    assert speed(100., 80.) == {"speedup_factor": 1.25, "wall_reduction_percent": 20.}
    assert speed(100., 125.) == {"speedup_factor": .8, "wall_reduction_percent": -25.}
    for value in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            speed(100., value)


def test_pass_alignment_rejects_partial_or_mismatched_spans():
    row = {"start_relative_update": 781, "optimizer_updates": 781, "sample_presentations": 99968, "wall_seconds": 100.}
    result = paired_spans([row], [{**row, "wall_seconds": 80.}])
    assert result["passes"][0]["pass"] == 2
    assert result["aggregate"]["speedup_factor"] == 1.25
    for altered in ({**row, "optimizer_updates": 780}, {**row, "start_relative_update": 1562}):
        with pytest.raises(ValueError):
            paired_spans([row], [altered])


def test_device_overlap_is_merged_without_counting_cpu_dispatch(tmp_path):
    events = [{"ph": "X", "cat": cat, "name": name, "ts": start, "dur": duration}
        for cat, name, start, duration in (("kernel", "k", 0, 10), ("gpu_memcpy", "copy", 5, 10),
            ("kernel", "k2", 20, 2), ("cpu_op", "aten::to", 0, 100))]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    result = trace_observations(path)
    assert result["device_activity_union_us"] == 17
    assert result["aten_dispatch_calls"] == {"aten::to": 1}


def test_predictions_require_exact_rows_targets_and_finite_values():
    old = {"source_idx": torch.tensor([10, 11]), "target_eV": torch.tensor([1., 2.]), "prediction_eV": torch.tensor([1.25, 2.25])}
    new = {**old, "prediction_eV": torch.tensor([1.125, 2.125])}
    assert prediction_comparison(old, new)["optimized_minus_original_mae_eV"] == -.125
    for changed in ({**new, "source_idx": torch.tensor([11, 10])}, {**new, "target_eV": torch.tensor([1., 3.])},
            {**new, "prediction_eV": torch.tensor([float("nan"), 2.])}):
        with pytest.raises(ValueError):
            prediction_comparison(old, changed)
