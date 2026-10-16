"""Synthetic transport checks; no native execution or historical role access."""
import json
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
                "cursor": {"epoch": 1, "next_batch": 0}, "optimizer_step": 2,
                "sample_presentations": 4}, source / "last_checkpoint.pt")
    (source / "canonical_trace.json").write_text(json.dumps({"observations": [
        {"event": "observation", "optimizer_step": 2, "sample_presentations": 4}]}))
    transport.create_resume_bundle(source, context, trajectory, bundle)
    return source, bundle, context, trajectory, arm


def test_roundtrip_retains_source_and_only_restores_artifacts(bundle_case, tmp_path):
    source, bundle, context, trajectory, _ = bundle_case
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    report = transport.validate_resume_bundle(bundle, context=context, trajectory=trajectory)
    assert report["status"] == "TRANSPORT_VERIFIED"
    destination = tmp_path / "restored"
    assert transport.restore_resume_bundle(bundle, destination)["status"] == "RESTORED"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(ValueError, match="fresh output"):
        transport.restore_resume_bundle(bundle, destination)


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


def test_spec_validation_uses_registered_family_hook(bundle_case, monkeypatch):
    _, bundle, context, trajectory, arm = bundle_case
    from molgap import experiment_execution, family_training_runner
    observed = []
    monkeypatch.setattr(experiment_execution, "training_adapter",
                        lambda declaration: SimpleNamespace(module="molgap.family_training_runner"))
    monkeypatch.setattr(family_training_runner, "validate_screen_resume",
                        lambda **options: observed.append(options) or {"synthetic": True})
    spec = SimpleNamespace(identity=context["spec_identity"], to_dict=lambda: {
        "arms": [arm], "prospective": {"arms": [{"arm_id": arm["arm_id"],
            "trajectory_id": trajectory["trajectory_id"]}]}})
    result = transport.validate_resume_bundle(bundle, spec, context=context, trajectory=trajectory)
    assert result["status"] == "RESUME_VERIFIED"
    assert result["owner"] == {"synthetic": True}
    assert len(observed) == 1 and observed[0]["arm_id"] == "reference"


def test_safe_loader_has_one_owner():
    from molgap.training_reproducibility import safe_cpu_torch_load
    assert transport.safe_cpu_torch_load is safe_cpu_torch_load


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
