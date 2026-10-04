"""Frozen recovery guards with synthetic tensors; no worker or accelerator work."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from molgap.experiment_family_workflow import TargetIdentityBinding


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("fault", ["context", "cursor", "file_hash", "float64_target", "manifest_pin", "success", "runtime_fingerprint"])
def test_recovery_rejects_bad_retained_binding_before_worker(tmp_path, monkeypatch, fault):
    import molgap.experiment_execution as execution
    import molgap.experiment_family_workflow as family
    from molgap.experiment_spec import ExperimentSpec

    path = Path(__file__).resolve().parents[1] / "platforms/kaggle/resume_frozen_k1_arm.py"
    loader = importlib.util.spec_from_file_location("synthetic_frozen_recovery", path)
    adapter = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(adapter)
    source, package, recovery = tmp_path / "source", tmp_path / "package", tmp_path / "recovery"
    package.mkdir()
    (package / "experiment_spec.json").write_text("{}")
    context = SimpleNamespace(spec_identity="synthetic-spec", arm_id="strong", run_reference="synthetic/run",
                              training_recipe_sha256="c" * 64, to_dict=lambda: {"source": "frozen"})
    target = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    expected = {"epochs": 40, "optimizer_steps": 31240, "sample_presentations": 3998720,
                "target_sha256": hashlib.sha256(target.numpy().astype("<f4", copy=False).tobytes()).hexdigest()}
    _write(source / "recipe.json", {"acceptance_requirements": expected, "row_order_fingerprint": "a" * 64})
    recovery.mkdir()
    acceptance = tmp_path / "acceptance"
    manifest_path = acceptance / "target_manifest.json"
    _write(manifest_path, {"target_encoding": "little-endian-float32-contiguous-raw-bytes",
                           "development_target_sha256": expected["target_sha256"]})
    plan_path = acceptance / "plan.json"
    _write(plan_path, {"format": "molgap-family-acceptance-plan-v1", "spec_identity": context.spec_identity,
                      "arms": [{"arm_id": context.arm_id, "expected": expected,
                                "contract": {"sha256": context.training_recipe_sha256},
                                "reference_artifacts": {"target_manifest": {
                                    "path": "target_manifest.json", "sha256": _sha(manifest_path)}}}]})
    target_identity = TargetIdentityBinding(acceptance.absolute(), "plan.json", _sha(plan_path))
    checkpoint = {"context": context.to_dict(), "cursor": {"epoch": 39, "next_batch": 0,
                   "sampler_order_sha256": "a" * 64}, "optimizer_step": 30459, "sample_presentations": 3898752}
    if fault == "context":
        checkpoint["context"] = {"source": "changed"}
    elif fault == "cursor":
        checkpoint["cursor"]["next_batch"] = 1
    torch.save(checkpoint, recovery / "last_checkpoint.pt")
    torch.save({"weight": torch.tensor([1.0])}, recovery / "selected_model.pt")
    torch.save({"target_eV": target.double() if fault == "float64_target" else target},
               recovery / "development_predictions.pt")
    _write(recovery / "canonical_trace.json", {"observations": [{"event": "observation", "epoch_or_pass": epoch,
           "checkpoint_identity": "sha256:" + _sha(recovery / "last_checkpoint.pt")} for epoch in range(1, 40)]})
    _write(recovery / "canonical_trace.json.context.json", {"context": context.to_dict()})
    names = ["last_checkpoint.pt", "selected_model.pt", "development_predictions.pt",
             "canonical_trace.json", "canonical_trace.json.context.json"]
    _write(recovery / "resume_binding.json", {
        "spec_identity": context.spec_identity, "package_identity": "b" * 64, "arm_id": context.arm_id,
        "start_epoch": 39, "end_epoch": 40, "original_run": context.run_reference,
        "original_version": 1, "runtime_fingerprint": "frozen-runtime",
        "target_identity": {"plan_path": "plan.json", "plan_sha256": target_identity.plan_sha256},
        "files": {name: _sha(recovery / name) for name in names},
    })
    if fault == "file_hash":
        with (recovery / "last_checkpoint.pt").open("ab") as stream:
            stream.write(b"changed")
    elif fault == "manifest_pin":
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    launch_path = tmp_path / "launch.json"
    _write(launch_path, {"expected_package_identity": "b" * 64, "jobs": [],
                        "account": "synthetic", "run_reference": context.run_reference,
                        "target_identity": {"plan_path": "plan.json", "plan_sha256": target_identity.plan_sha256}})
    monkeypatch.setattr(ExperimentSpec, "from_json", lambda _text: SimpleNamespace(identity=context.spec_identity))
    monkeypatch.setattr(execution, "validate_execution_plan", lambda _spec, _jobs: [
        {"arm_id": context.arm_id, "module": "molgap.k1_screen_training", "recipe": "recipe.json", "device": 1}])
    monkeypatch.setattr(family.RunContext, "for_training", lambda *args, **kwargs: context)
    # Plan availability is covered by workflow staging tests; retain the actual
    # pinned digest implementation here to exercise target dtype/hash rejection.
    monkeypatch.setattr(TargetIdentityBinding, "from_acceptance_plan", lambda *args, **kwargs: target_identity)
    worker = Mock(side_effect=AssertionError("Worker must not start"))
    monkeypatch.setattr(adapter.subprocess, "run", worker)
    gpu = Mock(side_effect=AssertionError("GPU must not be queried"))
    monkeypatch.setattr(torch.cuda, "device_count", gpu)

    if fault in {"success", "runtime_fingerprint"}:
        phases = []
        def run_worker(command, **kwargs):
            phase = command[command.index("--phase") + 1]
            destination = Path(command[command.index("--output") + 1])
            phases.append(phase)
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
            if phase == "preflight":
                _write(destination / "runtime_manifest.json", {
                    "runtime_fingerprint": "changed-runtime" if fault == "runtime_fingerprint" else "frozen-runtime"})
                return SimpleNamespace(returncode=0)
            _write(destination / "output_manifest.json", {"progress": {
                key: expected[key] for key in ("epochs", "optimizer_steps", "sample_presentations")}})
            # The unchanged original worker fails its default f64 inspection.
            return SimpleNamespace(returncode=1)
        monkeypatch.setattr(adapter.subprocess, "run", run_worker)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "synthetic Tesla T4")
        def inspect(_root, **kwargs):
            if kwargs.get("target_identity") is None:
                return {"status": "BLOCKED", "blockers": ["Development target identity mismatch"]}
            assert kwargs["target_identity"] is target_identity
            return {"status": "MECHANICALLY_VERIFIED", "blockers": []}
        monkeypatch.setattr(family, "inspect_output", inspect)
        if fault == "runtime_fingerprint":
            with pytest.raises(ValueError, match="runtime differs"):
                adapter.resume_one_arm(source_root=source, package_dir=package, input_root=tmp_path,
                                       launch_path=launch_path, output=tmp_path / "output")
            assert phases == ["preflight"]
            assert not (tmp_path / "output/recovery_state.json").exists()
        else:
            adapter.resume_one_arm(source_root=source, package_dir=package, input_root=tmp_path,
                                   launch_path=launch_path, output=tmp_path / "output")
            assert phases == ["preflight", "train"]
            state = json.loads((tmp_path / "output/recovery_state.json").read_text())
            assert state["status"] == "TRAINING_COMPLETE"
            assert state["acceptance_status"] == "MECHANICALLY_VERIFIED"
            assert state["additional_epochs"] == 1
            assert state["original_version"] == 1
            assert state["assigned_device_seconds"] >= 0
            assert "prior segment excluded" in state["device_cost_scope"]
            assert state["scientific_acceptance"] == "NOT_EVALUATED"
        return

    with pytest.raises(ValueError, match="disagree|hash mismatch|original float32"):
        adapter.resume_one_arm(source_root=source, package_dir=package, input_root=tmp_path,
                               launch_path=launch_path, output=tmp_path / "output")

    worker.assert_not_called()
    gpu.assert_not_called()
