"""Synthetic-only bounds, instrumentation neutrality and timing summaries."""
import json
from copy import deepcopy

import pytest
import torch

from molgap.gptrans_execution_diagnostic import (
    configuration, observe_step, phase_device, phases, read_diagnostic,
    summarize_probes, tensor_deltas, variants_for_arm,
)


def test_frozen_bounds_and_device_exchange(tmp_path):
    for kind in ("attribution", "timing"):
        record = configuration(kind)
        path = tmp_path / kind / "diagnostic.json"
        path.parent.mkdir()
        path.write_text(json.dumps(record))
        assert read_diagnostic(tmp_path, kind) == record
        record["measured_steps"] += 1
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            read_diagnostic(tmp_path, kind)
    assert phases(configuration("timing")) == (
        "preflight", "diagnostic-0", "diagnostic-1", "diagnostic-2", "diagnostic-profile")
    assert [phase_device(i, "diagnostic-1") for i in (0, 1)] == [1, 0]
    assert [phase_device(i, "diagnostic-0") for i in (0, 1)] == [0, 1]
    assert set(variants_for_arm("original") + variants_for_arm("optimized")) == {
        "native", "prepared", "foreach_optimizer", "foreach_ema", "optimized"}


def test_server_research_policy_preserves_explicit_roles_objective_and_segments():
    from molgap.family_training_policy import mode_policy
    bounded = mode_policy("gptrans_t", "degree_aggregate_correspondence_ema999")
    assert bounded.segment_passes == 10 and bounded.extra_development_weights == ("live",)
    assert bounded.objective is not None and bounded.after_update is not None
    assert bounded.train_metric_unit == "eV" and bounded.qualify_resources is not None


def test_observer_preserves_updates_and_restores_after_failure():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.1), torch.nn.Linear(4, 1))
    first = deepcopy(model)
    batch = torch.ones(128, 2)
    results = []
    original_clip = torch.nn.utils.clip_grad_norm_
    for current, observe in ((first, False), (model, True)):
        optimizer = torch.optim.AdamW(current.parameters(), lr=.001)
        torch.manual_seed(42)
        def step(observer=None):
            from molgap.family_training_core import optimizer_update
            return optimizer_update(current, optimizer, lambda: (current(batch).abs().mean(), None),
                clip_norm=1., check_finite=True, observer=observer)
        if observe:
            with observe_step(current) as (evidence, observer):
                assert torch.nn.utils.clip_grad_norm_ is original_clip
                step(observer)
            assert set(evidence) == {"prediction", "raw_gradients", "clipped_gradients", "gradient_norm"}
        else:
            step()
        results.append((deepcopy(current.state_dict()), deepcopy(optimizer.state_dict()), torch.get_rng_state()))
    assert tensor_deltas(results[0], results[1])["equal"]
    with pytest.raises(RuntimeError):
        with observe_step(model):
            raise RuntimeError("synthetic failure")
    assert torch.nn.utils.clip_grad_norm_ is original_clip
    assert not model._forward_hooks


def test_tensor_comparison_rejects_schema_and_nonfinite():
    old = {"x": torch.tensor([1., 2.])}
    assert tensor_deltas(old, {"x": torch.tensor([1., 2.25])})["maximum_absolute_delta"] == .25
    for changed in ({"x": torch.ones(3)}, {"y": torch.ones(2)}, {"x": torch.tensor([1., float("nan")])}):
        with pytest.raises(ValueError):
            tensor_deltas(old, changed)


def test_timing_summary_keeps_all_repeats_and_profile_separate(tmp_path):
    for phase in ("diagnostic-0", "diagnostic-1", "diagnostic-2", "diagnostic-profile"):
        for arm, seconds in (("original", 100.), ("optimized", 80.)):
            path = tmp_path / arm / phase / "output_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "COMPLETE", "phase": phase,
                "context": {"arm_id": arm}, "physical_device": str(phase_device(0 if arm == "original" else 1, phase)),
                "variants": [{"measured_wall_seconds": seconds}]}))
    result = summarize_probes(tmp_path, configuration("timing"))
    summary = json.loads((tmp_path / result["path"]).read_bytes())
    assert len(summary["paired_timings"]) == 3
    assert summary["median_paired_speedup"] == 1.25
    assert summary["paired_timings"][1]["physical_devices"] == {"original": "1", "optimized": "0"}
    saved = (tmp_path / result["path"]).read_bytes()
    assert summarize_probes(tmp_path, configuration("timing"), publish=False) == summary
    assert (tmp_path / result["path"]).read_bytes() == saved


@pytest.mark.parametrize("damage", [None, "ema", "missing_slot", "wrong_step", "rng"])
def test_saved_probe_state_checks_complete_optimizer_ema_and_rng(damage):
    from molgap.gptrans_execution_probe_acceptance import inspect_variant_state
    from molgap.experiment_family_workflow import tensor_safe_rng_state
    from molgap.training_reproducibility import capture_rng_state
    rng = tensor_safe_rng_state(capture_rng_state())
    rng["cuda"] = [torch.get_rng_state()]
    state = {"model": {"weight": torch.ones(2)}, "ema": {"weight": torch.ones(2)},
        "optimizer": {"param_groups": [{"params": [0]}], "state": {
            0: {"step": torch.tensor(10.), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}}},
        "scheduler": {"epoch": 0}, "rng_state": rng}
    if damage == "ema":
        state["ema"]["weight"][0] = float("nan")
    elif damage == "missing_slot":
        state["optimizer"]["state"].clear()
    elif damage == "wrong_step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(9.)
    elif damage == "rng":
        state["rng_state"]["cuda"] = []
    if damage:
        with pytest.raises(ValueError):
            inspect_variant_state(state, 10)
    else:
        inspect_variant_state(state, 10)
