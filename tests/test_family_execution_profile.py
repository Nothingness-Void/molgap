"""CPU observation must leave the shared training trajectory exactly intact."""
from copy import deepcopy
import json

import pytest
import torch

from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig
from test_family_training_session import assert_equal, events, make_session


def session(family="neural_atom_k1", log=None):
    return make_session(family, development=False, schedule_kind="cosine_step",
        events=events() if log is None else log)


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_profile_preserves_full_state_rng_callbacks_and_stops_capture(tmp_path, family):
    baseline_log, observed_log = events(), events()
    baseline = session(family, baseline_log)
    baseline_progress = baseline.run()
    baseline_state = deepcopy(baseline.state_dict())
    observed = session(family, observed_log)
    backward, clip = torch.autograd.backward, torch.nn.utils.clip_grad_norm_
    instance_keys = set(vars(observed.kernel.model))
    with FamilyExecutionProfiler(observed, tmp_path / "profile",
            ProfileConfig(warmup_steps=1, active_steps=2, module_paths=("0",))) as profile:
        assert observed.run() == baseline_progress
    assert_equal(baseline_state, observed.state_dict())
    assert_equal(baseline_log, observed_log)
    assert torch.autograd.backward is backward
    assert torch.nn.utils.clip_grad_norm_ is clip
    assert set(vars(observed.kernel.model)) == instance_keys
    report = profile.report
    assert report["status"] == "COMPLETE"
    assert report["observed_updates"] == 5 and report["captured_updates"] == 2
    assert report["capture_start_optimizer_step"] == 1
    assert report["device_events_observed"] is False
    assert report["operators_by_self_device"] == []
    assert all(row["self_device_us"] is None for row in report["operators_by_self_cpu"])
    phases = {row["name"]: row for row in report["phases_by_host_self_time"]}
    for name in ("forward", "backward", "optimizer", "ema", "module/0", "loader_wait"):
        assert phases[name]["calls"] == 2
    assert phases["checkpoint_publish"]["calls"] == 1
    assert all(row["host_total_seconds"] >= row["host_self_seconds"] >= 0 for row in phases.values())
    assert (tmp_path / "profile/trace.json").is_file()
    loaded = json.loads((tmp_path / "profile/profile.json").read_text())
    assert loaded["trace_sha256"] == report["trace_sha256"]


def test_partial_warmup_consumes_no_extra_batches(tmp_path):
    observed = session()
    with FamilyExecutionProfiler(observed, tmp_path / "partial",
            ProfileConfig(warmup_steps=3, active_steps=2)) as profile:
        progress = observed.run(should_pause=lambda: True)
    assert progress.status == "paused" and progress.optimizer_step == 1
    assert profile.report["status"] == "PARTIAL"
    assert profile.report["captured_updates"] == 0
    assert profile.report["capture_wall_seconds"] is None
    assert profile.report["trace_sha256"] is None


def test_final_update_checkpoint_is_inside_capture(tmp_path):
    observed = session()
    with FamilyExecutionProfiler(observed, tmp_path / "last",
            ProfileConfig(warmup_steps=3, active_steps=2)) as profile:
        observed.run()
    phases = {row["name"]: row for row in profile.report["phases_by_host_self_time"]}
    assert profile.report["status"] == "COMPLETE"
    assert phases["checkpoint_publish"]["calls"] == 2
    assert phases["selected_publish"]["calls"] == 1


def test_failed_checkpoint_propagates_and_restores_instrumentation(tmp_path):
    observed = session()
    def failure(*args):
        raise OSError("checkpoint fault")
    observed.checkpoint = failure
    backward, clip = torch.autograd.backward, torch.nn.utils.clip_grad_norm_
    with pytest.raises(OSError, match="checkpoint fault"):
        with FamilyExecutionProfiler(observed, tmp_path / "failed",
                ProfileConfig(warmup_steps=0, active_steps=4)) as profile:
            observed.run()
    assert observed.closed
    assert profile.report["status"] == "ERROR"
    assert profile.report["captured_updates"] == 2
    assert observed.checkpoint is failure
    assert torch.autograd.backward is backward
    assert torch.nn.utils.clip_grad_norm_ is clip
    # Lock and patched functions must also be usable after the error.
    replacement = session()
    with FamilyExecutionProfiler(replacement, tmp_path / "retry", ProfileConfig(0, 1)):
        replacement.run(should_pause=lambda: True)


def test_invalid_module_and_existing_output_do_not_leave_wrappers(tmp_path):
    observed = session()
    original = torch.autograd.backward
    with pytest.raises(AttributeError):
        with FamilyExecutionProfiler(observed, tmp_path / "invalid",
                ProfileConfig(module_paths=("missing",))):
            pytest.fail("Must reject before running")
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(FileExistsError):
        with FamilyExecutionProfiler(observed, tmp_path):
            pytest.fail("Must not replace an existing output")
    assert torch.autograd.backward is original
    observed.close()


def test_nested_profile_rejected_without_damaging_outer(tmp_path):
    outer, inner = session(), session()
    with FamilyExecutionProfiler(outer, tmp_path / "outer", ProfileConfig(0, 1)) as profile:
        with pytest.raises(RuntimeError, match="Only one"):
            with FamilyExecutionProfiler(inner, tmp_path / "inner"):
                pytest.fail("Cannot nest profilers")
        outer.run(should_pause=lambda: True)
    inner.close()
    assert profile.report["status"] == "COMPLETE"


@pytest.mark.parametrize("fault", [False, True])
def test_layer_hooks_preserve_native_cache_dispatch_and_cleanup(tmp_path, monkeypatch, fault):
    from test_gptrans_topology_reuse import small_local_model, synthetic_batch
    from molgap import gptrans_capacity as local
    from molgap.gptrans_execution_backend import prepare_layout, prepared_forward
    observed = session("gptrans_t")
    model = small_local_model(local.BondLocalGPTrans).eval()
    observed.kernel.model = model
    batch = prepare_layout(synthetic_batch())
    block = model.blocks[0]
    before = block.forward.__func__
    hooks = (dict(block._forward_pre_hooks), dict(block._forward_hooks))
    original, calls = local._flat_bond_targets, []
    def counted(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(local, "_flat_bond_targets", counted)
    with torch.no_grad():
        expected = prepared_forward(model, batch)
    assert len(calls) == 1
    calls.clear()
    def fail(*args):
        raise RuntimeError("layer prehook fault")
    failure = block.register_forward_pre_hook(fail) if fault else None
    try:
        with FamilyExecutionProfiler(observed, tmp_path / "layer",
                ProfileConfig(0, 1, module_paths=("blocks.0",))) as profiler:
            assert block.forward.__func__ is before
            profiler._begin_batch()
            if fault:
                with pytest.raises(RuntimeError, match="layer prehook fault"):
                    prepared_forward(model, batch)
                assert not profiler._stack
                failure.remove()
                calls.clear()
            with torch.no_grad():
                assert torch.equal(prepared_forward(model, batch), expected)
            assert len(calls) == 1
            assert not profiler._stack
        assert block.forward.__func__ is before
        assert (dict(block._forward_pre_hooks), dict(block._forward_hooks)) == hooks
        assert not model._molgap_substrate._active
    finally:
        if failure is not None:
            failure.remove()
        observed.close()


@pytest.mark.parametrize("kwargs", [
    {"warmup_steps": -1}, {"active_steps": 0}, {"active_steps": True},
    {"record_shapes": 1}, {"row_limit": 0}, {"module_paths": ("0", "0")},
])
def test_invalid_profile_dimensions(kwargs):
    with pytest.raises(ValueError):
        ProfileConfig(**kwargs)
def test_profile_retention_pins_trace_and_native_proof_without_environment_files(tmp_path):
    from molgap.family_execution_profile import profile_artifact_refs
    from molgap.training_reproducibility import sha256_file

    selected = ["execution_profile_receipt.json", "runtime_certificate.json",
                "profiles/step_0/profile.json", "profiles/step_0/trace.json",
                "profile_qualification/plain-state.pt", "repeat-a.pt"]
    for name in selected + ["environment/site.py", "last_checkpoint.pt", "unrelated.json"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    records = profile_artifact_refs(tmp_path)
    assert {item["path"] for item in records} == set(selected)
    assert all(item["sha256"] == sha256_file(tmp_path / item["path"]) for item in records)
    (tmp_path / "profiles/step_0/trace.json").write_bytes(b"changed")
    assert profile_artifact_refs(tmp_path) != records
