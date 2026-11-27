"""Synthetic transport checks; no native execution or historical role access."""
import json
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import experiment_resume as transport
from molgap.experiment_retention import validate_allocation_ledger
from molgap.screen_policy import canonical_fingerprint
from test_experiment_workflow import _candidate_pair_spec


@pytest.fixture
def bundle_case(tmp_path):
    arm = {"arm_id": "reference", "family": {"name": "neural_atom_k1", "version": "3"},
           "training": {"recipe": {"sha256": "f" * 64}}}
    context = {
        "experiment_id": "synthetic-resume", "logical_run_id": "synthetic-run",
        "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
        "spec_identity": "b" * 64, "family_name": arm["family"]["name"],
        "family_version": arm["family"]["version"], "source_commit": "c" * 40,
        "source_archive_sha256": "d" * 64, "package_identity": "e" * 64,
        "training_recipe_sha256": "f" * 64, "platform": "local", "account": "local",
        "run_reference": "synthetic/resume", "platform_version": None,
    }
    trajectory = {
        "trajectory_id": "synthetic-trajectory", "record_mode": "prospective",
        "state_at_start": {"source_commit": context["source_commit"],
                           "source_config_identity": context["arm_identity"]},
        "actions": [{"action_id": "A1", "run_ids": ["synthetic-run:reference"],
                     "attempt_ids": ["attempt1"]}],
        "decision": {"outcome": "ACTIVE"},
    }
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    torch.save({"context": context, "model": {"weight": torch.ones(1)},
                "cursor": {"epoch": 1, "next_batch": 0,
                           "pending_absolute": [torch.tensor(.25)],
                           "weight_state": {"clean": {"buffers": {"running_mean": torch.ones(3)}}}},
                "optimizer_step": 2,
                "sample_presentations": 4}, source / "last_checkpoint.pt")
    (source / "canonical_trace.json").write_text(json.dumps({"observations": [
        {"event": "observation", "optimizer_step": 2, "sample_presentations": 4}]}))
    transport.create_resume_bundle(source, context, trajectory, bundle)
    return source, bundle, context, trajectory, arm


def test_roundtrip_retains_source_and_only_restores_artifacts(bundle_case, tmp_path, monkeypatch):
    source, bundle, context, trajectory, _ = bundle_case
    from molgap.family_training_recipe import aux_ema_names
    _, _, model_role, pred_role, model_file, pred_file, _ = aux_ema_names(.9999)
    torch.save({"model": {"weight": torch.ones(1)}, "weights": "ema-0.9999"}, source / model_file)
    torch.save({"prediction_eV": torch.ones(3)}, source / pred_file)
    bundle = tmp_path / "auxiliary-bundle"
    transport.create_resume_bundle(source, context, trajectory, bundle)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    report = transport.validate_resume_bundle(bundle, context=context, trajectory=trajectory)
    assert report["status"] == "TRANSPORT_VERIFIED"
    assert all(report["manifest"]["artifacts"][role]["required"] for role in (model_role, pred_role))
    assert "pending_absolute" not in report["manifest"]["cursor"]
    assert "weight_state" not in report["manifest"]["cursor"]
    # Pre-fix bundles retained this empty JSON field; remain readable unchanged.
    legacy = json.loads((bundle / transport.RESUME_BUNDLE_FILE).read_bytes())
    legacy["cursor"]["pending_absolute"] = []
    (bundle / transport.RESUME_BUNDLE_FILE).write_bytes(transport._json_bytes(legacy))
    assert transport.validate_resume_bundle(bundle, context=context)["status"] == "TRANSPORT_VERIFIED"
    destination = tmp_path / "restored"
    assert transport.restore_resume_bundle(bundle, destination)["status"] == "RESTORED"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(ValueError, match="fresh output"):
        transport.restore_resume_bundle(bundle, destination)

    copyfile = transport.shutil.copyfile
    corrupted = []

    def corrupt_after_copy(source, target):
        result = copyfile(source, target)
        with Path(target).open("ab") as stream:
            stream.write(b"changed during copy")
        corrupted.append(Path(target))
        return result

    failed_output = tmp_path / "failed-copy"
    with monkeypatch.context() as patch:
        patch.setattr(transport.shutil, "copyfile", corrupt_after_copy)
        with pytest.raises(ValueError, match="bytes changed during restoration"):
            transport.restore_resume_bundle(bundle, failed_output)
    assert len(corrupted) == 1
    assert not any(path.is_file() for path in failed_output.rglob("*"))
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_shared_resume_requires_explicit_float32_target_binding(bundle_case, tmp_path, monkeypatch):
    from molgap import experiment_execution, family_training_runner, family_training_session
    from molgap.experiment_family_workflow import TargetIdentityBinding, tensor_digest
    from molgap.family_training_recipe import aux_ema_names
    from molgap.research_memory import trace

    source, _, context, trajectory, arm = bundle_case
    context = {**context, "run_reference": "local/resume"}
    spec = SimpleNamespace(identity=context["spec_identity"], to_dict=lambda: {"arms": [arm],
        "prospective": {"arms": [{"arm_id": arm["arm_id"], "trajectory_id": trajectory["trajectory_id"]}]}})
    monkeypatch.setattr(experiment_execution, "training_adapter",
                        lambda _: SimpleNamespace(module=family_training_runner.__name__))
    label, _, _, _, model_file, pred_file, metric = aux_ema_names(.9999)
    target = torch.tensor([0.1], dtype=torch.float32)
    indices = torch.tensor([0])
    expected = {"development_rows": 1, "source_idx_sha256": tensor_digest(indices, role="source_idx"),
                "target_sha256": transport.hashlib.sha256(target.numpy().astype("<f4").tobytes()).hexdigest()}
    recipe = SimpleNamespace(dataset=SimpleNamespace(development=True), auxiliary_ema_decays=(.9999,),
                             optimizer_steps=4, selection="best-development-ema", identity="recipe")
    monkeypatch.setattr(family_training_runner, "read_recipe",
                        lambda _: ({"acceptance_requirements": expected}, recipe))
    monkeypatch.setattr(family_training_runner, "validate_screen_recipe", lambda *_: None)
    monkeypatch.setattr(family_training_session, "validate_session_resume",
                        lambda *_: SimpleNamespace(status="paused", pass_index=1,
                                                   optimizer_step=2, sample_presentations=4))
    monkeypatch.setattr(trace, "validate_canonical_trace", lambda value: value)
    checkpoint_path = source / "last_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["context"] = context
    checkpoint["cursor"]["auxiliary_selections"] = {label: {"selected_step": 2, "best_development_mae_eV": 0.}}
    torch.save(checkpoint, checkpoint_path)
    torch.save({"context": context, "model": checkpoint["model"], "weights": "ema-" + label,
                "selection": recipe.selection, "optimizer_step": 2, "development_mae_eV": 0.}, source / model_file)
    torch.save({"context": context, "source_idx": indices, "target_eV": target,
                "prediction_eV": target}, source / pred_file)
    trace_path = source / "canonical_trace.json"
    curve = json.loads(trace_path.read_bytes())
    curve["trajectory_id"] = trajectory["trajectory_id"]
    curve["observations"][0][metric] = 0.
    trace_path.write_text(json.dumps(curve), encoding="utf-8")
    bundle = tmp_path / "float32-bundle"
    transport.create_resume_bundle(source, context, trajectory, bundle)

    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(json.dumps({"development_target_sha256": expected["target_sha256"],
        "target_encoding": "little-endian-float32-contiguous-raw-bytes"}), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"format": "molgap-family-same-run-acceptance-plan-v1",
        "spec_identity": spec.identity, "arms": [{"arm_id": arm["arm_id"], "expected": expected,
        "contract": {"sha256": context["training_recipe_sha256"]}, "target_manifest": {
            "path": manifest_path.name, "sha256": trace.file_digest(manifest_path)}}]}), encoding="utf-8")
    binding = TargetIdentityBinding(tmp_path, plan_path.name, trace.file_digest(plan_path))
    with pytest.raises(ValueError, match="Development target identity mismatch"):
        transport.validate_resume_bundle(bundle, spec, target_identity=None)
    assert transport.validate_resume_bundle(bundle, spec, target_identity=binding)["status"] == "RESUME_VERIFIED"
    assert transport.restore_resume_bundle(bundle, tmp_path / "restored", spec,
                                           target_identity=binding)["status"] == "RESTORED"


def test_bundle_binds_captured_allocation_sidecar_during_atomic_refresh(bundle_case, tmp_path, monkeypatch):
    source, _, ctx, trajectory, _ = bundle_case
    ledger = source / "allocation_ledger.json"
    before, after = b'{"generation":1}', b'{"generation":2}'
    ledger.write_bytes(before)
    original_read = Path.read_bytes
    captured = []
    def refresh_after_read(path):
        payload = original_read(path)
        if path == ledger and not captured:
            captured.append(payload)
            replacement = ledger.with_name(".allocation_ledger.next.json")
            replacement.write_bytes(after)
            os.replace(replacement, ledger)
        return payload
    monkeypatch.setattr(Path, "read_bytes", refresh_after_read)
    destination = tmp_path / "snapshot"
    manifest = transport.create_resume_bundle(source, ctx, trajectory, destination)
    assert captured == [before] and ledger.read_bytes() == after
    pointer = manifest["cost_segments"][0]
    assert pointer["sha256"] == transport.hashlib.sha256(before).hexdigest()
    assert (destination / pointer["path"]).read_bytes() == before
    assert transport.validate_resume_bundle(destination, context=ctx)["status"] == "TRANSPORT_VERIFIED"


@pytest.mark.parametrize("defect", ["hash", "cursor", "escape", "context", "trajectory"])
def test_transport_rejects_unbound_or_changed_evidence(bundle_case, defect):
    _, bundle, context, trajectory, _ = bundle_case
    manifest_path = bundle / "resume_bundle.json"
    manifest = json.loads(manifest_path.read_bytes())
    if defect == "hash":
        manifest["artifacts"]["checkpoint"]["sha256"] = "0" * 64
    elif defect == "cursor":
        manifest["cursor"]["optimizer_step"] += 1
    elif defect == "escape":
        manifest["artifacts"]["checkpoint"]["path"] = "../checkpoint.pt"
    elif defect == "context":
        context = dict(context, source_archive_sha256="0" * 64)
    else:
        trajectory = dict(trajectory, trajectory_id="different")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        transport.validate_resume_bundle(bundle, context=context, trajectory=trajectory)


@pytest.mark.parametrize("cross_volume", [False, True])
def test_bundle_reuses_only_immutable_verified_payload(bundle_case, tmp_path, monkeypatch, cross_volume):
    source, previous, ctx, trajectory, _ = bundle_case
    destination = tmp_path / "shared"
    links, real_link = [], transport.os.link
    def link(origin, target):
        if origin.is_relative_to(previous):
            links.append((origin, target))
            if cross_volume:
                raise OSError(errno.EXDEV, "synthetic cross-volume boundary")
        return real_link(origin, target)
    monkeypatch.setattr(transport.os, "link", link)
    transport.create_resume_bundle(source, ctx, trajectory, destination, reuse_bundle=previous)
    assert links and all(origin.is_relative_to(previous) for origin, _ in links)
    retained = destination / "files/last_checkpoint.pt"
    assert os.path.samefile(previous / "files/last_checkpoint.pt", retained) is not cross_volume
    assert not os.path.samefile(source / "last_checkpoint.pt", retained)
    before = retained.read_bytes()
    (source / "last_checkpoint.pt").write_bytes(b"mutable producer changed")
    assert retained.read_bytes() == before
    assert transport.validate_resume_bundle(destination, context=ctx)["status"] == "TRANSPORT_VERIFIED"


def test_reuse_rejects_changed_payload_and_different_attempt(bundle_case, tmp_path):
    source, previous, ctx, trajectory, _ = bundle_case
    changed = dict(trajectory, actions=[{"action_id": "A1", "run_ids": ["synthetic-run:reference"],
                                       "attempt_ids": ["attempt2"]}])
    with pytest.raises(ValueError, match="trajectory"):
        transport.create_resume_bundle(source, ctx, changed, tmp_path / "different", reuse_bundle=previous)
    assert not (tmp_path / "different").exists()
    (previous / "files/last_checkpoint.pt").write_bytes(b"changed frozen payload")
    with pytest.raises(ValueError):
        transport.create_resume_bundle(source, ctx, trajectory, tmp_path / "unsafe", reuse_bundle=previous)
    assert not (tmp_path / "unsafe").exists()


@pytest.mark.parametrize("extra", ["partial", "unknown_file", "unknown_directory"])
def test_prune_preserves_partial_and_unlisted_chunk_evidence(bundle_case, tmp_path, extra):
    source, _, ctx, trajectory, _ = bundle_case
    parent = tmp_path / "chunks"
    older, newest = parent / "pass_01", parent / "pass_02"
    for folder in (older, newest):
        transport.create_resume_bundle(source, ctx, trajectory, folder)
    if extra == "partial":
        (older / "files/last_checkpoint.pt").unlink()
    elif extra == "unknown_file":
        (older / "unique_contract.json").write_text("unique evidence")
    else:
        (older / "unique_evidence").mkdir()
    report = transport.prune_duplicate_resume_bundles(parent, newest, keep_count=1, context=ctx)
    assert report["retained_unverified"][0]["chunk"] == "pass_01"
    assert not report["removed_duplicates"] and older.exists()


def test_prune_reports_interrupted_removal_and_preserves_partial_on_retry(bundle_case, tmp_path, monkeypatch):
    source, _, ctx, trajectory, _ = bundle_case
    parent = tmp_path / "chunks"
    older, newest = parent / "pass_01", parent / "pass_02"
    for folder in (older, newest):
        transport.create_resume_bundle(source, ctx, trajectory, folder)
    def interrupted(folder):
        (folder / "files/last_checkpoint.pt").unlink()
        raise PermissionError("synthetic interrupted removal")
    monkeypatch.setattr(transport.shutil, "rmtree", interrupted)
    for _ in range(2):
        report = transport.prune_duplicate_resume_bundles(parent, newest, keep_count=1, context=ctx)
        assert report["retained_unverified"][0]["chunk"] == "pass_01"
        assert not report["removed_duplicates"]
        assert older.exists()
    assert transport.validate_resume_bundle(newest, context=ctx)["status"] == "TRANSPORT_VERIFIED"


def test_prune_validates_new_bundle_before_any_deletion(bundle_case, tmp_path, monkeypatch):
    source, _, ctx, trajectory, _ = bundle_case
    parent = tmp_path / "chunks"
    older, newest = parent / "pass_01", parent / "pass_02"
    for folder in (older, newest):
        transport.create_resume_bundle(source, ctx, trajectory, folder)
    (newest / "files/last_checkpoint.pt").write_bytes(b"incomplete new publication")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe new bundle must not release deletion")
    monkeypatch.setattr(transport.shutil, "rmtree", forbidden)
    with pytest.raises(ValueError):
        transport.prune_duplicate_resume_bundles(parent, newest, keep_count=1, context=ctx)
    assert transport.validate_resume_bundle(older, context=ctx)["status"] == "TRANSPORT_VERIFIED"


@pytest.fixture
def ledger_case():
    spec = _candidate_pair_spec()
    arms = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    ledger = {
        "format": "molgap-allocation-ledger-v1", "spec_identity": spec.identity,
        "started_at": "2026-10-16T00:00:00+00:00", "observed_at": "2026-10-16T00:00:10+00:00",
        "status": "complete", "allocation_device_count": 2, "wall_seconds": 10.0,
        "allocated_device_seconds": 20.0, "unassigned_device_seconds": 0.0,
        "devices": [{"device": index, "hardware": "Tesla T4", "arm_id": arm,
                     "allocated_seconds": 10.0} for index, arm in enumerate(arms)],
        "scope": "Python bootstrap/runtime observation window; includes idle allocated devices",
        "provisioning_before_python_seconds": {"value": None, "status": "measurement_missing"},
        "queue_seconds": {"value": None, "status": "measurement_missing"},
        "prior_segments": [], "prior_unobserved_intervals": False,
    }
    return spec, ledger


def test_allocation_cost_units_remain_explicit(ledger_case):
    spec, ledger = ledger_case
    assert validate_allocation_ledger(ledger, spec) == ledger
    assert ledger["queue_seconds"]["value"] is None


@pytest.mark.parametrize("defect", ["identity", "device_count", "device_cost", "assignment", "queue"])
def test_allocation_ledger_rejects_unsupported_claims(ledger_case, defect):
    spec, ledger = ledger_case
    if defect == "identity":
        ledger["spec_identity"] = "0" * 64
    elif defect == "device_count":
        ledger["allocation_device_count"] = 1
    elif defect == "device_cost":
        ledger["allocated_device_seconds"] = 10.0
    elif defect == "assignment":
        ledger["devices"][1]["arm_id"] = ledger["devices"][0]["arm_id"]
    else:
        ledger["queue_seconds"] = {"value": 0.0, "status": "measured"}
    with pytest.raises(ValueError):
        validate_allocation_ledger(ledger, spec)
