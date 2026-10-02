"""Transport-only tests for incomplete per-arm resume bundles."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import numpy as np

from molgap import experiment_execution, gptrans_screen_workflow
from molgap.experiment_resume import (
    create_resume_bundle,
    restore_resume_bundle,
    validate_resume_bundle,
    safe_cpu_torch_load,
)
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import capture_rng_state


def _context(family="synthetic"):
    return {
        "experiment_id": "resume-experiment",
        "logical_run_id": "resume-run",
        "arm_id": "reference",
        "arm_identity": "a" * 64,
        "spec_identity": "b" * 64,
        "family_name": family,
        "family_version": "1",
        "source_commit": "c" * 40,
        "source_archive_sha256": "d" * 64,
        "package_identity": "e" * 64,
        "training_recipe_sha256": "f" * 64,
        "platform": "local",
        "account": "local",
        "run_reference": "local/resume",
        "platform_version": None,
    }


def _trajectory(context):
    return {
        "trajectory_id": "trajectory-resume",
        "record_mode": "prospective",
        "state_at_start": {
            "source_commit": context["source_commit"],
            "source_config_identity": context["arm_identity"],
        },
        "actions": [{"action_id": "A001", "run_ids": ["resume-run:reference"],
                      "attempt_ids": ["v1"]}],
        "decision": {"outcome": "ACTIVE"},
    }


def _source(tmp_path: Path, context):
    source = tmp_path / "native"
    source.mkdir(parents=True)
    checkpoint = {
        "context": context,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": [{}]},
        "rng_state": {"python": (3, (1,), None), "numpy": ("MT19937", [1], 0, 0, 0.0),
                       "torch": torch.ones(1, dtype=torch.uint8), "cuda": []},
        "cursor": {"epoch": 1, "next_batch": 0, "sampler_order_sha256": "1" * 64},
        "optimizer_step": 1,
        "sample_presentations": 2,
    }
    torch.save(checkpoint, source / "last_checkpoint.pt")
    (source / "canonical_trace.json").write_text(json.dumps({"format": "synthetic"}), encoding="utf-8")
    return source


def test_create_validate_and_restore_is_hash_bound(tmp_path):
    context = _context()
    source = _source(tmp_path, context)
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(_trajectory(context)), encoding="utf-8")
    bundle = tmp_path / "bundle"

    manifest = create_resume_bundle(source, context, trajectory_path, bundle)
    assert manifest["status"] == "incomplete"
    assert manifest["run"] == {"logical_run_id": "resume-run", "action_id": "A001",
                                "run_id": "resume-run:reference", "attempt_id": "v1"}
    assert validate_resume_bundle(bundle)["status"] == "TRANSPORT_VERIFIED"

    restored = tmp_path / "restored"
    assert restore_resume_bundle(bundle, restored)["status"] == "RESTORED"
    assert (restored / "last_checkpoint.pt").is_file()
    assert (restored / "canonical_trace.json").is_file()
    assert not (restored / "trajectory.json").exists()


def test_restore_rechecks_hash_after_copy_and_cleans_temporary_file(tmp_path, monkeypatch):
    import molgap.experiment_resume as transport

    context = _context()
    source = _source(tmp_path, context)
    bundle = tmp_path / "bundle"
    create_resume_bundle(source, context, _trajectory(context), bundle)
    manifest_bytes = (bundle / "resume_bundle.json").read_bytes()
    original_copyfile = transport.shutil.copyfile
    tampered = False

    def tamper_before_copy(source_path, destination_path, *args, **kwargs):
        nonlocal tampered
        if Path(source_path).name == "last_checkpoint.pt" and not tampered:
            tampered = True
            Path(source_path).write_bytes(Path(source_path).read_bytes() + b"tampered")
        return original_copyfile(source_path, destination_path, *args, **kwargs)

    monkeypatch.setattr(transport.shutil, "copyfile", tamper_before_copy)
    restored = tmp_path / "restored-after-tamper"
    with pytest.raises(ValueError, match="hash|artifact|bytes changed"):
        restore_resume_bundle(bundle, restored)
    assert tampered is True
    assert (bundle / "resume_bundle.json").read_bytes() == manifest_bytes
    assert not (restored / "last_checkpoint.pt").exists()
    assert not any(restored.rglob("*"))


def test_restore_rejects_concurrent_target_publication_without_overwrite(tmp_path, monkeypatch):
    import molgap.experiment_resume as transport

    context = _context()
    source = _source(tmp_path, context)
    bundle = tmp_path / "bundle"
    create_resume_bundle(source, context, _trajectory(context), bundle)
    original_link = transport.os.link
    concurrent_bytes = b"published-by-concurrent-writer"
    raced = False

    def publish_race(source_path, target_path, *args, **kwargs):
        nonlocal raced
        if Path(target_path).name == "last_checkpoint.pt" and not raced:
            raced = True
            Path(target_path).write_bytes(concurrent_bytes)
        return original_link(source_path, target_path, *args, **kwargs)

    monkeypatch.setattr(transport.os, "link", publish_race)
    restored = tmp_path / "restored-after-race"
    with pytest.raises(ValueError, match="conflict|overwrite|publication"):
        restore_resume_bundle(bundle, restored)
    assert raced is True
    assert (restored / "last_checkpoint.pt").read_bytes() == concurrent_bytes
    assert not any(path.name.endswith(".resume.tmp") for path in restored.rglob("*"))


def test_bundle_rejects_changed_prospective_bytes_and_restore_conflicts(tmp_path):
    context = _context()
    source = _source(tmp_path, context)
    trajectory = _trajectory(context)
    bundle = tmp_path / "bundle"
    create_resume_bundle(source, context, trajectory, bundle)

    changed = dict(trajectory, trajectory_id="changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        validate_resume_bundle(bundle, trajectory=changed)

    restored = tmp_path / "restored"
    restored.mkdir()
    (restored / "last_checkpoint.pt").write_bytes(b"conflict")
    with pytest.raises(ValueError, match="fresh output"):
        restore_resume_bundle(bundle, restored)


def test_completed_native_outputs_cannot_be_exported(tmp_path):
    context = _context()
    source = _source(tmp_path, context)
    (source / "output_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed family output"):
        create_resume_bundle(source, context, _trajectory(context), tmp_path / "bundle")

    source = _source(tmp_path / "second", context)
    (source / "completion_manifest.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="Completed GPTrans"):
        create_resume_bundle(source, context, _trajectory(context), tmp_path / "bundle2")


def test_bundle_rejects_ambiguous_attempt(tmp_path):
    context = _context()
    source = _source(tmp_path, context)
    trajectory = _trajectory(context)
    trajectory["actions"][0]["attempt_ids"] = ["v1", "v2"]
    with pytest.raises(ValueError, match="exact prospective action"):
        create_resume_bundle(source, context, trajectory, tmp_path / "bundle")


def test_safe_cpu_loader_handles_gptrans_numpy_rng_without_pickle_execution(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"rng_state": {"numpy": np.random.RandomState(42).get_state()},
                "model": {"weight": torch.ones(2)}}, path)
    loaded = safe_cpu_torch_load(path)
    assert loaded["model"]["weight"].device.type == "cpu"
    assert isinstance(loaded["rng_state"]["numpy"][1], np.ndarray)


def test_safe_cpu_loader_rejects_object_numpy_arrays(tmp_path):
    path = tmp_path / "unsafe-checkpoint.pt"
    torch.save({"bad": np.array(["callable"], dtype=object)}, path)
    with pytest.raises(ValueError, match="safe CPU tensor artifact|object NumPy array"):
        safe_cpu_torch_load(path)


class _SyntheticSpec:
    logical_run_id = "resume-run"

    def __init__(self, family="gptrans_t", initialization=None):
        self.family = family
        self.initialization = initialization

    def to_dict(self):
        arm = {
            "arm_id": "reference", "family": {"name": self.family,
                                                   "version": "1" if self.family == "gptrans_t" else "2"},
        }
        if self.initialization is not None:
            arm["initialization"] = dict(self.initialization)
        return {"logical_run_id": self.logical_run_id, "arms": [arm]}


class _SyntheticAdapter:
    def mode(self, arm):
        return "reference"


def _gptrans_owner_fixture(tmp_path, monkeypatch):
    """Build a CPU-only native prefix for the owner state gate."""
    monkeypatch.setattr(experiment_execution, "training_adapter", lambda arm: _SyntheticAdapter())
    monkeypatch.setattr(gptrans_screen_workflow.owner, "CHECKPOINT_FORMAT", "test-checkpoint")
    monkeypatch.setattr(gptrans_screen_workflow.owner, "RUN_FORMAT", "test-trace")
    monkeypatch.setattr(gptrans_screen_workflow.owner, "BATCHES_PER_EPOCH", 1)
    monkeypatch.setattr(gptrans_screen_workflow.owner, "PHYSICAL_BATCH", 2)
    monkeypatch.setattr(gptrans_screen_workflow.owner, "EPOCHS", 3)
    monkeypatch.setattr(gptrans_screen_workflow.owner, "_scientific_fields",
                        lambda mode: {"mode": mode})
    monkeypatch.setattr(gptrans_screen_workflow.owner, "_ema_decay", lambda mode: 0.9)
    root = tmp_path / "gptrans-owner"
    root.mkdir()
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
                   "platform_id": "kaggle2", "accelerator": "NVIDIA Tesla T4",
                   "precision": "fp32", "tf32_enabled": False,
                   "deterministic_algorithms": True, "physical_batch_per_device": 128,
                   "tail_batch_policy": "drop_last", "software_fingerprint": "2" * 64,
                   "determinism_fingerprint": "3" * 64,
                   "calibration_fixture_sha256": "4" * 64,
                   "calibration_output_sha256": "5" * 64,
                   "calibration_checks_passed": True, "runtime_fingerprint": "1" * 64}
    certificate_id = canonical_fingerprint(certificate)
    runtime = {"runtime_fingerprint": certificate["runtime_fingerprint"]}
    trace = {"format": "test-trace", "rows": [{
        "epoch": 0, "train_mae_eV": 1.0, "development_mae_eV": 1.0,
        "best_development_mae_eV": 1.0, "best_epoch": 0,
        "learning_rate": 0.001, "elapsed_seconds": 1.0,
        "optimizer_steps": 1, "sample_presentations": 2,
    }]}
    trace_bytes = json.dumps(trace, sort_keys=True).encode()
    (root / "trace.json").write_bytes(trace_bytes)
    (root / "runtime_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
    (root / "runtime_manifest.json").write_text(json.dumps(runtime), encoding="utf-8")
    preflight = {"accepted": True, "variant": "reference",
                 "runtime_certificate_id": certificate_id,
                 "runtime_certificate": certificate, "runtime_manifest": runtime,
                 "source_archive_sha256": "d" * 64, "source_commit": "c" * 40}
    (root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    rng_state = capture_rng_state()
    rng_state["cuda"] = [torch.ones(1, dtype=torch.uint8)]
    checkpoint = {
        "format": "test-checkpoint", "variant": "reference", "epoch": 0,
        "model": {"weight": torch.ones(1)}, "optimizer": {"state": {}, "param_groups": [{}]},
        "scheduler": {"epoch": 0}, "ema": {"weight": torch.ones(1)},
        "trace": trace["rows"], "best_development_mae_eV": 1.0, "best_epoch": 0,
        "target_stats": {"mean_eV": 0.0, "sample_std_eV": 1.0},
        "runtime_certificate_id": certificate_id, "source_archive_sha256": "d" * 64,
        "rng_state": rng_state, "scientific_fields": {"mode": "reference"},
        "ema_decay": 0.9,
    }
    torch.save(checkpoint, root / "last_checkpoint.pt")
    torch.save({"format": "test-trace", "model_config": {"variant": "reference"},
                "model": {"weight": torch.ones(1)}, "target_stats": {"mean_eV": 0.0,
                "sample_std_eV": 1.0}, "epoch": 0, "source_archive_sha256": "d" * 64,
                "source_commit": "c" * 40, "runtime_certificate_id": certificate_id},
               root / "best_model.pt")
    torch.save({"prediction_eV": torch.ones(1), "target_eV": torch.ones(1),
                "source_idx": torch.zeros(1, dtype=torch.long),
                "official_validation_role_read": False, "test_dev_role_read": False,
                "test_challenge_role_read": False}, root / "development_predictions.pt")
    context = _context("gptrans_t")
    (root / "workflow_context.json").write_text(json.dumps(context), encoding="utf-8")
    artifacts = {"checkpoint": root / "last_checkpoint.pt", "trace": root / "trace.json",
                 "selected_model": root / "best_model.pt", "predictions": root / "development_predictions.pt"}
    sidecars = {"runtime": [root / "runtime_certificate.json", root / "runtime_manifest.json",
                             root / "preflight.json"],
                "provenance": [], "context": [root / "workflow_context.json"], "cost_segments": []}
    manifest = {"trajectory": {"id": "trajectory-resume"}}
    return root, _SyntheticSpec(), context, artifacts, sidecars, manifest, checkpoint


def test_gptrans_owner_rejects_missing_native_optimizer_state(tmp_path, monkeypatch):
    root, spec, context, artifacts, sidecars, manifest, checkpoint = _gptrans_owner_fixture(
        tmp_path, monkeypatch)
    checkpoint.pop("optimizer")
    torch.save(checkpoint, artifacts["checkpoint"])
    with pytest.raises(ValueError, match="checkpoint state is incomplete"):
        gptrans_screen_workflow.validate_screen_resume(
            spec, "reference", manifest, root, artifacts, sidecars,
            {"trajectory_id": "trajectory-resume"}, context=context)


def test_gptrans_owner_rejects_missing_cuda_rng_state(tmp_path, monkeypatch):
    root, spec, context, artifacts, sidecars, manifest, checkpoint = _gptrans_owner_fixture(
        tmp_path, monkeypatch)
    checkpoint["rng_state"]["cuda"] = []
    torch.save(checkpoint, artifacts["checkpoint"])
    with pytest.raises(ValueError, match="CUDA RNG schema"):
        gptrans_screen_workflow.validate_screen_resume(
            spec, "reference", manifest, root, artifacts, sidecars,
            {"trajectory_id": "trajectory-resume"}, context=context)


def test_gptrans_owner_rejects_changed_retained_certificate(tmp_path, monkeypatch):
    root, spec, context, artifacts, sidecars, manifest, _ = _gptrans_owner_fixture(
        tmp_path, monkeypatch)
    changed = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
               "runtime_fingerprint": "2" * 64}
    (root / "runtime_certificate.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="certificate identity mismatch"):
        gptrans_screen_workflow.validate_screen_resume(
            spec, "reference", manifest, root, artifacts, sidecars,
            {"trajectory_id": "trajectory-resume"}, context=context)


class _ContextObject:
    def __init__(self, value):
        self._value = dict(value)
        self.__dict__.update(self._value)

    def to_dict(self):
        return dict(self._value)


def test_gptrans_resume_preflight_reuses_exact_certificate_without_diagnostic(tmp_path, monkeypatch):
    root, spec, context, artifacts, sidecars, manifest, _ = _gptrans_owner_fixture(
        tmp_path, monkeypatch)
    context_object = _ContextObject(context)
    monkeypatch.setattr(gptrans_screen_workflow, "_validate_request",
                        lambda **kwargs: (context_object, {}))
    monkeypatch.setattr(gptrans_screen_workflow.owner, "validate_source_archive",
                        lambda *args: None)
    runtime = json.loads((root / "runtime_manifest.json").read_bytes())
    import molgap.training_reproducibility as reproducibility
    monkeypatch.setattr(reproducibility, "configure_fp32_determinism", lambda seed: {})
    monkeypatch.setattr(reproducibility, "build_runtime_manifest", lambda determinism: runtime)
    monkeypatch.setattr(gptrans_screen_workflow.owner, "run_preflight",
                        lambda *args, **kwargs: pytest.fail("resume must not run diagnostics"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Tesla T4")
    destination = tmp_path / "preflight"
    result = gptrans_screen_workflow.run_screen_resume_preflight(
        spec=spec, package_dir=tmp_path, expected_package_identity="e" * 64,
        arm_id="reference", mode="reference", recipe_path=tmp_path / "recipe.json",
        initial_state_path=tmp_path / "initial.pt", input_root=tmp_path,
        output=destination, account="local", run_reference="local/resume",
        trajectory_id="trajectory-resume", resume_output=root)
    assert result["resume_reused"] is True
    for name in ("runtime_manifest.json", "runtime_certificate.json", "preflight.json"):
        assert (destination / name).read_bytes() == (root / name).read_bytes()
    assert json.loads((destination / "workflow_context.json").read_bytes()) == context


def test_gptrans_resume_preflight_rejects_changed_current_runtime(tmp_path, monkeypatch):
    root, spec, context, artifacts, sidecars, manifest, _ = _gptrans_owner_fixture(
        tmp_path, monkeypatch)
    context_object = _ContextObject(context)
    monkeypatch.setattr(gptrans_screen_workflow, "_validate_request",
                        lambda **kwargs: (context_object, {}))
    monkeypatch.setattr(gptrans_screen_workflow.owner, "validate_source_archive",
                        lambda *args: None)
    import molgap.training_reproducibility as reproducibility
    monkeypatch.setattr(reproducibility, "configure_fp32_determinism", lambda seed: {})
    monkeypatch.setattr(reproducibility, "build_runtime_manifest",
                        lambda determinism: {"runtime_fingerprint": "9" * 64})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Tesla T4")
    with pytest.raises(ValueError, match="runtime fingerprint"):
        gptrans_screen_workflow.run_screen_resume_preflight(
            spec=spec, package_dir=tmp_path, expected_package_identity="e" * 64,
            arm_id="reference", mode="reference", recipe_path=tmp_path / "recipe.json",
            initial_state_path=tmp_path / "initial.pt", input_root=tmp_path,
            output=tmp_path / "preflight", account="local", run_reference="local/resume",
            trajectory_id="trajectory-resume", resume_output=root)


def _k1_owner_fixture(tmp_path, monkeypatch):
    """Build a CPU-only K1 family prefix with complete retained preflight evidence."""
    import hashlib
    from molgap import k1_screen_training
    from molgap.research_memory.trace import RMLTraceRecorder

    monkeypatch.setattr(experiment_execution, "training_adapter", lambda arm: _SyntheticAdapter())
    monkeypatch.setattr(k1_screen_training, "ROW_ORDER_FINGERPRINT", "1" * 64)
    monkeypatch.setattr(k1_screen_training, "EPOCHS", 3)
    monkeypatch.setattr(k1_screen_training, "STEPS_PER_EPOCH", 2)
    monkeypatch.setattr(k1_screen_training, "ROWS_PER_EPOCH", 4)
    monkeypatch.setattr(k1_screen_training, "validate_recipe", lambda *args, **kwargs: None)
    monkeypatch.setattr(k1_screen_training, "_validate_arm_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(k1_screen_training, "_validate_prospective", lambda *args, **kwargs: None)
    root = tmp_path / "k1-owner"
    root.mkdir()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(b"{}\n")
    initial_state_path = tmp_path / "initial.pt"
    initial_state_path.write_bytes(b"initial-state")
    context = _context("neural_atom_k1")
    context["family_version"] = "2"
    context["training_recipe_sha256"] = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    monkeypatch.setattr(k1_screen_training, "INITIAL_STATE_SHA256",
                        hashlib.sha256(initial_state_path.read_bytes()).hexdigest())
    context_object = _ContextObject(context)
    semantics = {
        "live_train_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
                               "role_identity": "train", "weights": "live", "direction": "minimize"},
        "live_dev_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
                             "role_identity": "dev", "weights": "live", "direction": "minimize"},
        "ema_dev_metric": None,
    }
    trace_path = root / "canonical_trace.json"
    recorder = RMLTraceRecorder(trace_path, trajectory_id="trajectory-resume",
        run_id="resume-run:reference:downstream", metric_semantics=semantics)
    recorder.append_observation(optimizer_step=2, sample_presentations=4,
        epoch_or_pass=1, learning_rate=0.001, live_train_metric=1.0,
        live_dev_metric=1.0, wall_time_seconds=1.0)
    rng_state = capture_rng_state()
    rng_state["cuda"] = [torch.ones(1, dtype=torch.uint8)]
    checkpoint = {"context": context, "model": {"weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": [{}]}, "scheduler": {"last_epoch": 1},
        "rng_state": rng_state,
        "cursor": {"epoch": 1, "next_batch": 0, "sampler_order_sha256": "1" * 64},
        "optimizer_step": 2, "sample_presentations": 4}
    torch.save(checkpoint, root / "last_checkpoint.pt")
    torch.save({"context": context, "model": {"weight": torch.ones(1)}, "epoch": 1,
                "optimizer_step": 2, "weights": "live", "development_mae_eV": 1.0},
               root / "selected_model.pt")
    torch.save({"context": context, "prediction_eV": torch.ones(1),
                "target_eV": torch.ones(1), "source_idx": torch.zeros(1, dtype=torch.long)},
               root / "development_predictions.pt")
    runtime = {"runtime_fingerprint": "r" * 64}
    provenance = k1_screen_training._runtime_provenance(
        context_object, recipe_path, initial_state_path, runtime, "reference")
    architecture = {"accepted": True, "repeatability": {"accepted": True},
        "resume_roundtrip": {"accepted": True}, "zero_initialization_delta": 0.0,
        "maximum_overhead_fraction": 0.25,
        "synchronized_step_overhead_fraction": 0.1}
    certificate = {"status": "accepted", "runtime_fingerprint": runtime["runtime_fingerprint"],
                   "provenance_sha256": canonical_fingerprint(provenance),
                   "architecture_sha256": canonical_fingerprint(architecture),
                   "calibration_checks_passed": True,
                   "accelerator": "NVIDIA Tesla T4"}
    (root / "runtime_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (root / "runtime_manifest.json").write_text(json.dumps(runtime), encoding="utf-8")
    (root / "runtime_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
    (root / "architecture_preflight.json").write_text(json.dumps(architecture), encoding="utf-8")
    (root / "canonical_trace.json.context.json").write_text(
        json.dumps({"context": context, "stage_id": "downstream"}), encoding="utf-8")

    class FakeRunContext:
        @staticmethod
        def for_training(*args, **kwargs):
            return context_object

    monkeypatch.setattr("molgap.experiment_family_workflow.RunContext", FakeRunContext)
    monkeypatch.setattr(k1_screen_training, "build_runtime_manifest", lambda determinism: runtime)
    monkeypatch.setattr(k1_screen_training, "configure_fp32_determinism", lambda seed: {})
    return root, _SyntheticSpec(
        "neural_atom_k1",
        {"kind": "frozen_state", "seed": k1_screen_training.SEED,
         "state_sha256": k1_screen_training.INITIAL_STATE_SHA256},
    ), context_object, recipe_path, initial_state_path, runtime


def test_k1_resume_preflight_reuses_exact_certificate_without_diagnostic(tmp_path, monkeypatch):
    from molgap import k1_screen_training
    root, spec, context, recipe_path, initial_state_path, runtime = _k1_owner_fixture(
        tmp_path, monkeypatch)
    monkeypatch.setattr(k1_screen_training, "run_screen_preflight",
                        lambda *args, **kwargs: pytest.fail("resume must not run diagnostics"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Tesla T4")
    destination = tmp_path / "k1-preflight"
    result = k1_screen_training.run_screen_resume_preflight(
        spec=spec, package_dir=tmp_path, expected_package_identity="e" * 64,
        arm_id="reference", mode="reference", recipe_path=recipe_path,
        initial_state_path=initial_state_path, input_root=tmp_path, output=destination,
        account="local", run_reference="local/resume", trajectory_id="trajectory-resume",
        resume_output=root)
    assert result["resume_reused"] is True
    for name in ("runtime_provenance.json", "runtime_manifest.json",
                 "runtime_certificate.json", "architecture_preflight.json"):
        assert (destination / name).read_bytes() == (root / name).read_bytes()


def test_k1_resume_preflight_rejects_changed_current_runtime(tmp_path, monkeypatch):
    from molgap import k1_screen_training
    root, spec, context, recipe_path, initial_state_path, runtime = _k1_owner_fixture(
        tmp_path, monkeypatch)
    monkeypatch.setattr(k1_screen_training, "build_runtime_manifest",
                        lambda determinism: {"runtime_fingerprint": "x" * 64})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA Tesla T4")
    with pytest.raises(ValueError, match="runtime preflight identity changed"):
        k1_screen_training.run_screen_resume_preflight(
            spec=spec, package_dir=tmp_path, expected_package_identity="e" * 64,
            arm_id="reference", mode="reference", recipe_path=recipe_path,
            initial_state_path=initial_state_path, input_root=tmp_path,
            output=tmp_path / "k1-preflight", account="local", run_reference="local/resume",
            trajectory_id="trajectory-resume", resume_output=root)


def test_k1_owner_rejects_missing_cuda_rng_state(tmp_path, monkeypatch):
    from molgap import k1_screen_training
    root, spec, context, recipe_path, initial_state_path, runtime = _k1_owner_fixture(
        tmp_path, monkeypatch)
    checkpoint_path = root / "last_checkpoint.pt"
    checkpoint = safe_cpu_torch_load(checkpoint_path)
    checkpoint["rng_state"]["cuda"] = []
    torch.save(checkpoint, checkpoint_path)
    artifacts = {"checkpoint": checkpoint_path, "trace": root / "canonical_trace.json",
                 "selected_model": root / "selected_model.pt",
                 "predictions": root / "development_predictions.pt"}
    sidecars = {"runtime": [root / "runtime_manifest.json", root / "runtime_certificate.json",
                             root / "architecture_preflight.json"],
                "provenance": [root / "runtime_provenance.json"],
                "context": [root / "canonical_trace.json.context.json"], "cost_segments": []}
    with pytest.raises(ValueError, match="CUDA RNG schema"):
        k1_screen_training.validate_screen_resume(
            spec, "reference", {"trajectory": {"id": "trajectory-resume"}}, root,
            artifacts, sidecars, {"trajectory_id": "trajectory-resume"}, context=context.to_dict())
