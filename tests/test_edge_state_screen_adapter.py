"""Metadata and mock delegation only; no model training/inference."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap.edge_state_screen_adapter import bind_edge_state_screen
from molgap.experiment_spec import ExperimentSpec
from molgap.research_memory.trace import file_digest


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    import molgap.edge_state_screen_adapter as module
    from molgap.constants import REPO_ROOT
    raw = json.loads((Path(REPO_ROOT) / "docs/operations/examples/edge_state_v1.json").read_text())
    raw["schema_version"] = "molgap-experiment-spec-v2"
    k1 = copy.deepcopy(raw["arms"][0])
    k1["arm_id"] = "edge-k1"
    k1["scientific_role"] = "candidate"
    k1["addons"] = [{"name": "neural_atom_k1", "version": "1", "config": {}, "source_sha256": "a" * 64}]
    k1["addon_semantics"] = "ordered"
    raw["arms"].append(k1)
    raw["prospective"] = {"arms": [
        {"arm_id": arm["arm_id"], "trajectory_id": f"TC-{arm['arm_id']}",
         "plan_spec_ref": f"experiments/synthetic/{arm['arm_id']}/plan.json", "plan_spec_sha256": "a" * 64,
         "output": f"experiments/synthetic/{arm['arm_id']}/rml"} for arm in raw["arms"]]}
    spec = ExperimentSpec(raw)
    monkeypatch.setattr(module, "verify_experiment_source_package", lambda _: {
        "spec_identity": spec.identity, "archive_sha256": "b" * 64, "source_commit": "c" * 40})
    manifest, contract = tmp_path / "manifest.json", tmp_path / "contract.json"
    manifest.write_text('{"fixture":"no graphs"}')
    contract.write_text('{"fixture":"not a release"}')
    return dict(spec=spec, package_dir=tmp_path / "package", output_root=tmp_path / "outputs",
                graph_manifest=manifest, expected_graph_manifest_sha256=file_digest(manifest),
                training_contract=contract, expected_training_contract_sha256=file_digest(contract),
                runtime_fingerprint="d" * 64, train_rows=100000, target_mean_eV=5.0, target_std_eV=1.0)


@pytest.mark.parametrize("arm_id,variant,layers", [
    ("edge-state-9", "edge_state", 9), ("edge-state-6", "edge_state_depth", 6),
    ("edge-k1", "neural_atom_k1", 9)])
def test_series_share_same_binding_core(inputs, arm_id, variant, layers):
    screen = bind_edge_state_screen(**inputs, arm_id=arm_id)
    assert screen.binding.variant == variant
    assert screen.binding.num_layers == layers
    assert screen.binding.physical_batch_size == 128
    assert screen.trajectory_id == f"TC-{arm_id}"
    assert screen.output == inputs["output_root"] / arm_id
    frozen = json.loads((screen.output / "screen_binding.json").read_text())
    assert frozen["binding"] == asdict(screen.binding)
    assert bind_edge_state_screen(**inputs, arm_id=arm_id) == screen


def test_each_arm_isolated_and_resume_rejects_changed_contract(inputs):
    first = bind_edge_state_screen(**inputs, arm_id="edge-state-9")
    second = bind_edge_state_screen(**inputs, arm_id="edge-state-6")
    assert first.run_id != second.run_id
    assert first.output != second.output
    inputs["runtime_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="resume identity changed"):
        bind_edge_state_screen(**inputs, arm_id="edge-state-9")


@pytest.mark.parametrize("key", ["expected_graph_manifest_sha256", "expected_training_contract_sha256"])
def test_changed_metadata_never_bound(inputs, key):
    inputs[key] = "0" * 64
    with pytest.raises(ValueError, match="artifact SHA"):
        bind_edge_state_screen(**inputs, arm_id="edge-state-9")
    assert not inputs["output_root"].exists()


def test_wrong_package_rejected(inputs, monkeypatch):
    monkeypatch.setattr("molgap.edge_state_screen_adapter.verify_experiment_source_package", lambda _: {
        "spec_identity": "0" * 64})
    with pytest.raises(ValueError, match="differs"):
        bind_edge_state_screen(**inputs, arm_id="edge-state-9")


def test_orphan_outputs_not_adopted(inputs):
    output = inputs["output_root"] / "edge-state-9"
    output.mkdir(parents=True)
    (output / "last.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="reconciliation"):
        bind_edge_state_screen(**inputs, arm_id="edge-state-9")


def test_model_step_eval_sampler_checkpoint_delegate_without_execution(inputs, monkeypatch):
    import molgap.edge_state_screen_adapter as module
    screen = bind_edge_state_screen(**inputs, arm_id="edge-state-9")
    calls = {}
    for name in ("construct_bound_model", "EpochPermutationBatchSampler", "normalized_gap_step",
                 "evaluate_development", "save_checkpoint", "restore_checkpoint"):
        calls[name] = Mock(return_value=object())
        monkeypatch.setattr(module.core, name, calls[name])
    model, optimizer, scheduler, batch, source_idx = (object() for _ in range(5))
    screen.construct_model()
    calls["construct_bound_model"].assert_called_once_with(screen.spec, screen.binding)
    screen.sampler(epoch=7, start_batch=3)
    calls["EpochPermutationBatchSampler"].assert_called_once_with(100000, 128, seed=42, epoch=7, start_batch=3)
    screen.step(model, optimizer, batch, clip_norm=1.0)
    calls["normalized_gap_step"].assert_called_once_with(model, optimizer, batch, screen.binding, clip_norm=1.0)
    screen.evaluate(model, batch, expected_source_idx=source_idx)
    calls["evaluate_development"].assert_called_once_with(model, batch, screen.binding, expected_source_idx=source_idx)
    cursor = dict(epoch=7, batch_offset=3, optimizer_steps=5470, sample_presentations=700160, sampler_state={})
    screen.save_checkpoint(model, optimizer, scheduler, **cursor)
    calls["save_checkpoint"].assert_called_once_with(screen.output / "last.pt", screen.binding, model, optimizer, scheduler, **cursor)
    screen.restore_checkpoint(model, optimizer, scheduler, expected_sha256="a" * 64)
    calls["restore_checkpoint"].assert_called_once_with(screen.output / "last.pt", "a" * 64, screen.binding,
                                                        model, optimizer, scheduler, loader_generator=None)


def test_trace_identity_and_device_time_semantics_reused(inputs):
    screen = bind_edge_state_screen(**inputs, arm_id="edge-state-9")
    semantics = {"live_train_metric": None, "live_dev_metric": None, "ema_dev_metric": None}
    recorder = screen.recorder(metric_semantics=semantics, device_time_semantics="sum_over_devices")
    recorder.append_observation(optimizer_step=1, sample_presentations=128,
                               cumulative_device_time_seconds=2.0)
    assert screen.recorder(metric_semantics=semantics, device_time_semantics="sum_over_devices").record == recorder.record
    assert recorder.record["trajectory_id"] == screen.trajectory_id
    assert recorder.record["run_id"] == screen.run_id
    # Metadata recording alone must not claim terminal/replay acceptance.
    assert recorder.record["observations"][0]["event"] == "observation"
