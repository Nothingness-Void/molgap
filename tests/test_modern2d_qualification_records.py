"""Synthetic saved metadata only; no checkpoints, network or training.

The fixture replaces byte pins explicitly. Production preparation exposes no
pin override and remains bound to the independently inspected V4 attempt.
"""
import hashlib
import json
from pathlib import Path

import pytest

from molgap import modern2d_qualification_records as records
from molgap.common.wire import canonical_fingerprint
from molgap.research_memory.finalize import finalize
from molgap.research_memory.roles import validate_observed_role_truth
from molgap.research_memory.terminal_wiring import inspect_trace_retention_evidence
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.v5_common import validate_v5_evidence_envelope


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))
    return file_digest(path)


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    from test_terminal_trace_closure import setup_mock_repo

    root, inspection = tmp_path / "repo", tmp_path / "inspection"
    root.mkdir()
    setup_mock_repo(root)
    base = root / records.BASE
    save(base / "authority.json", {"purpose": "EXECUTION_QUALIFICATION_ONLY"})
    save(base / "role_plan.json", {"roles_opened": ["train"], "development_metrics": False})
    observation = {"kernel": records.KERNEL, "kernel_id": 139279865, "version": 1,
        "status": "KernelWorkerStatus.COMPLETE", "failure_message": "", "private": True,
        "entry_bytes_equal": True, "machine_shape": "NvidiaTeslaT4"}
    save(inspection / "platform_observation.json", observation)
    audit = {"result": "PASS_METADATA_BINDING_PENDING_FULL_COST", "run": observation,
        "source_pins": {"source_archive_sha256": "a" * 64, "package_identity": "b" * 64},
        "checks": {"data_version_roles": {"status": "PASS", "opened_roles": ["train_100k"],
            "common_dev_permit_frozen_not_opened_for_metrics": True, "development_rows": 0,
            "development_metrics": False, "head_to_head_comparison": False}},
        "cost": {"status": "INSUFFICIENT_EVIDENCE", "native_cost_fully_accepted": False,
            "overlap_not_additive": True, "full_allocation_seconds": None,
            "full_allocation_device_seconds": None}, "arms": {}}
    verification = {"status": "PASS", "cuda_initialized_before_load": False,
        "cuda_initialized_after_load": False, "arms": {}}
    execution = {"status": "complete", "phase": "sealed", "formal_training_exposure": 0,
        "native_execution_released": False, "scientific_acceptance": "NOT_EVALUATED",
        "spec_identity": records.SPEC, "launch_sha256": records.LAUNCH, "workers": {}}
    files, transport = [], []
    out = inspection / "outputs/qualification"

    def artifact(name, value):
        digest = save(out / name, value)
        files.append({"path": name, "sha256": digest, "bytes": (out / name).stat().st_size})
        return digest

    for device, (arm, interface) in enumerate(records.ARMS.items()):
        tid, run_id = f"TC-modern2d-qualification-{arm}-s42-v4", f"{records.RUN}:{arm}:downstream"
        frozen = json.loads((root / "experiments/ref_exp/trajectory.json").read_bytes())
        frozen.update(trajectory_id=tid, owner="server", record_mode="prospective", track="C")
        frozen["hypothesis"].update(hypothesis_id=f"H-{tid}", observed_deficiency="native unqualified",
            changed_mechanism="versioned execution", cheapest_falsifier="six scratch updates",
            expected_native_cost_ref="cost-test-1", decision_changed_if_positive="qualification only",
            decision_changed_if_negative="close execution attempt", supporting_evidence_ids=["ev-ref-1"],
            alternative_explanations=["runtime"], related_closed_family_ids=["F-ref"])
        frozen["state_at_start"].update(source_commit=records.SOURCE, source_config_identity="synthetic",
            reference_ids=[], prior_evidence_ids=["ev-ref-1"], role_snapshot_refs=[f"{records.BASE}/role_plan.json"],
            budget_snapshot_ref=f"{records.BASE}/authority.json")
        frozen["actions"] = [{"action_id": "A001", "run_ids": [run_id], "type": "execution_diagnostic",
            "source_commit": "0" * 40, "cost_event_ids": [], "attempt_ids": [f"{arm}-v4"], "evidence_refs": []}]
        frozen["result"] = {"evidence_ids": [], "evidence_refs": []}
        frozen["decision"].update(outcome="ACTIVE", decision_ref="experiments/ref_exp/decision.md")
        save(base / arm / "rml/trajectory.json", frozen)
        recipe = {"family": interface[0], "family_version": interface[1], "seed": 42,
            "batch_size": 128, "train_rows": 100000, "development_rows": 0,
            "optimizer_steps": 2, "dataset": {"development": False}}
        recipe_hash = save(base / arm / "contract.json", recipe)
        context = {"logical_run_id": records.RUN, "arm_id": arm, "family_name": interface[0],
            "family_version": interface[1], "source_commit": records.SOURCE, "spec_identity": records.SPEC,
            "arm_identity": "synthetic",
            "run_reference": records.KERNEL, "platform_version": None, "account": "kaseichou",
            **audit["source_pins"], "training_recipe_sha256": recipe_hash}
        q = {"run_context": context, "recipe_identity": "c" * 64,
            "train_permit": {"role": "train_100k", "version_identity": "d" * 64, "manifest_sha256": "e" * 64},
            "development_permit": {"role": "common_dev", "version_identity": "d" * 64}}
        identity = canonical_fingerprint(q)
        provenance = {"context": context, "qualification_context": q, "recipe_identity": "c" * 64,
            "initialization": {"state_sha256": "1" * 64}}
        certificate = {"status": "QUALIFICATION_VERIFIED", "scope": "QUALIFICATION_ONLY",
            "formal_training_exposure": 0, "scratch_optimizer_updates": 6, "native_execution_released": False,
            "scientific_acceptance": "NOT_EVALUATED", "qualification_context_identity": identity,
            "probe_recipe_identity": "c" * 64, "provenance_sha256": canonical_fingerprint(provenance), "proofs": []}
        checked = {"initial_state": {"model_state_sha256": "1" * 64}, "proofs": {}}
        for i, name in enumerate(("repeat-a.pt", "repeat-b.pt", "resumed.pt", "resumed-first.pt")):
            # These are hashes supplied by a verifier, not files to be opened.
            digest = hashlib.sha256(f"{arm}/{name}".encode()).hexdigest()
            pointer = f"{arm}/scratch/{name}"
            files.append({"path": pointer, "sha256": digest, "bytes": 123})
            transport.append({"path": f"qualification/{pointer}", "sha256": digest, "state": "RETAINED"})
            proof = {"path": name, "sha256": digest, "state_fingerprint": "2" * 64}
            state = {"status": "paused" if i == 3 else "complete", "optimizer_step": 1 if i == 3 else 2,
                "sample_presentations": 128 if i == 3 else 256, "next_batch": 1 if i == 3 else 2,
                "optimizer_step_values": [1 if i == 3 else 2], "development": False, "ema": False,
                "cuda_rng_count": 1, "state_fingerprint": "2" * 64, "model_state_sha256": "3" * 64}
            if i == 3:
                certificate["resume_origin"], checked["resume_origin"] = proof, state
            else:
                certificate["proofs"].append(proof)
                checked["proofs"][name] = state
        resources = {"scope": "QUALIFICATION_ONLY", "physical_batch_size": 128,
            "scratch_optimizer_updates": 6, "scratch_sample_presentations": 768, "allocated_device_seconds": None}
        certificate["resource_qualification_sha256"] = artifact(f"{arm}/scratch/resource_qualification.json", resources)
        artifact(f"{arm}/scratch/runtime_provenance.json", provenance)
        cert_hash = artifact(f"{arm}/scratch/runtime_certificate.json", certificate)
        artifact(f"{arm}/scratch/worker_result.json", {"status": "QUALIFICATION_VERIFIED", "device": device,
            "launch_sha256": records.LAUNCH, "context_identity": identity,
            "native_execution_released": False, "certificate_sha256": cert_hash})
        execution["workers"][arm] = {"device": device, "returncode": 0, "status": "QUALIFICATION_VERIFIED"}
        verification["arms"][arm] = checked
        audit["arms"][arm] = {"context_identity": identity, "binding": "PASS", "physical_batch_size": 128,
            "seed": 42, "scratch_optimizer_updates": 6, "formal_training_exposure": 0}
    artifact("execution_report.json", execution)
    artifact("allocation_ledger.json", {"status": "complete", "spec_identity": records.SPEC,
        "allocation_device_count": 2, "allocated_device_seconds": 660})
    save(out / "qualification_manifest.json", {"format": "molgap-modern2d-qualification-output-v1",
        "status": "complete", "scope": "QUALIFICATION_ONLY", "native_execution_released": False,
        "scientific_acceptance": "NOT_EVALUATED", "spec_identity": records.SPEC,
        "launch_sha256": records.LAUNCH, "files": files})
    save(inspection / "checkpoint_hash_acceptance.json", {"kernel": records.KERNEL, "receipt_version": 1,
        "platform_version_verified_by_output_api": True, "files": transport})
    save(inspection / "luna_states/checkpoint_verification.json", verification)
    save(inspection / "scratchinspection/luna_metadata/native_certificate_audit.json", audit)
    save(inspection / "outputs/bootstrap_allocation.json", {"status": "complete", "phase": "runtime_returned",
        "full_allocation_seconds": None, "full_allocation_device_seconds": None})
    monkeypatch.setattr(records, "_INPUT_PINS", {ref: file_digest(inspection / ref) for ref in records._INPUT_PINS})
    monkeypatch.setattr(records, "_REPO_PINS", {ref: file_digest(root / ref) for ref in records._REPO_PINS})
    return {"repo_root": root, "inspection_root": inspection, "output": "experiments/retained/results/qualification_terminal",
        "finalized_at": "2026-10-18T00:00:00+00:00"}


def test_prepares_idempotent_schema_valid_no_train_with_unknown_cost(metadata):
    result = records.prepare_qualification_terminal(**metadata)
    root = metadata["repo_root"]
    assert not list(root.rglob("rml_finalized"))
    assert records.prepare_qualification_terminal(**metadata)["status"] == "ALREADY_PREPARED"
    for arm in result["arms"].values():
        package = json.loads((root / arm["terminal"]).read_bytes())
        validate_v5_evidence_envelope(package["evidence"], repo_root=root)
        validate_observed_role_truth([package["role_use"]], package["roles"])
        assert inspect_trace_retention_evidence(package["evidence"]) == (False, None)
        assert "trace_manifest" not in package and "action_replay" not in package
        assert package["decision"]["outcome"] == "NO_TRAIN"
        assert package["evidence"]["outcome"]["artifact_status"] == "accepted"
        assert all(v == {"status": "measurement_missing", "value": None}
                   for v in package["costs"][0]["measurement"].values())
        assert {r["role_name"] for r in package["roles"]} == {"train_100k"}
        assert package["role_use"]["common_dev"] == "untouched"
        acceptance = json.loads((root / package["acceptance_ref"]).read_bytes())
        assert acceptance["noReplay"] and not acceptance["native_training_release"]
        for ref, digest in package["artifact_hashes"].items():
            assert not Path(ref).is_absolute() and file_digest(root / ref) == digest
        # Publication is allowed only inside this temporary synthetic repository.
        receipt = finalize(root, arm["trajectory"], arm["terminal"])
        assert receipt["trace_status"] == "unavailable"
        assert finalize(root, arm["trajectory"], arm["terminal"])["status"] == "ALREADY_FINALIZED"
    assert not list((root / metadata["output"]).rglob("*.pt"))


@pytest.mark.parametrize("target", ["platform_observation.json", "luna_states/checkpoint_verification.json",
    "outputs/qualification/ppgt_author_v2/scratch/runtime_certificate.json"])
def test_tamper_fails_before_writes(metadata, target):
    path = metadata["inspection_root"] / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        records.prepare_qualification_terminal(**metadata)
    assert not (metadata["repo_root"] / metadata["output"]).exists()


@pytest.mark.parametrize("target,field,value,error", [
    ("platform_observation.json", "version", 2, "platform observation"),
    ("platform_observation.json", "version", True, "platform observation"),
    ("platform_observation.json", "status", "KernelWorkerStatus.RUNNING", "platform observation"),
    ("luna_states/checkpoint_verification.json", "status", "FAIL", "independent verification"),
])
def test_rebound_bad_observation_still_rejected(metadata, monkeypatch, target, field, value, error):
    path = metadata["inspection_root"] / target
    payload = json.loads(path.read_bytes())
    payload[field] = value
    pins = {**records._INPUT_PINS, target: save(path, payload)}
    monkeypatch.setattr(records, "_INPUT_PINS", pins)
    with pytest.raises(ValueError, match=error):
        records.prepare_qualification_terminal(**metadata)


@pytest.mark.parametrize("field,value,error", [
    ("opened_roles", ["train_100k", "common_dev"], "role access"),
    ("opened_roles", ["official_validation"], "role access"),
    ("development_metrics", True, "role access"),
])
def test_permit_is_not_observed_access(metadata, monkeypatch, field, value, error):
    ref = "scratchinspection/luna_metadata/native_certificate_audit.json"
    path = metadata["inspection_root"] / ref
    payload = json.loads(path.read_bytes())
    payload["checks"]["data_version_roles"][field] = value
    monkeypatch.setattr(records, "_INPUT_PINS", {**records._INPUT_PINS, ref: save(path, payload)})
    with pytest.raises(ValueError, match=error):
        records.prepare_qualification_terminal(**metadata)


def test_original_prospective_tamper_and_no_clobber(metadata):
    prepared = records.prepare_qualification_terminal(**metadata)
    terminal = metadata["repo_root"] / prepared["arms"]["ppgt_author_v2"]["terminal"]
    terminal.write_bytes(b"user-owned change")
    with pytest.raises(ValueError, match="PREPARATION_CONFLICT"):
        records.prepare_qualification_terminal(**metadata)
    assert terminal.read_bytes() == b"user-owned change"
    trajectory = metadata["repo_root"] / prepared["arms"]["esa_native_v3"]["trajectory"]
    trajectory.write_bytes(trajectory.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        records.prepare_qualification_terminal(**metadata)


def test_root_escape_and_canonical_destination_rejected(metadata):
    for output in ("../escape", f"{records.BASE}/ppgt_author_v2/rml", "research_memory/derived"):
        with pytest.raises(ValueError):
            records.prepare_qualification_terminal(**{**metadata, "output": output})


def test_terminal_has_existing_schema_keys_only(metadata):
    result = records.prepare_qualification_terminal(**metadata)
    schema_path = Path(__file__).resolve().parents[1] / "research_memory/schemas/terminal-package-v1.schema.json"
    schema = json.loads(schema_path.read_bytes())
    for arm in result["arms"].values():
        terminal = json.loads((metadata["repo_root"] / arm["terminal"]).read_bytes())
        assert set(schema["required"]) <= set(terminal) <= set(schema["properties"])


@pytest.mark.parametrize("mutation,error", [("missing_cost_to_zero", "must be null"),
    ("permit_to_commondev_access", "without matching observed event")])
def test_existing_finalizer_guards_missing_cost_and_observed_roles(metadata, mutation, error):
    prepared = records.prepare_qualification_terminal(**metadata)
    arm = prepared["arms"]["ppgt_author_v2"]
    terminal_path = metadata["repo_root"] / arm["terminal"]
    terminal = json.loads(terminal_path.read_bytes())
    if mutation == "missing_cost_to_zero":
        terminal["costs"][0]["measurement"]["device_hours"]["value"] = 0
    else:
        terminal["role_use"]["common_dev"] = "consumed"
    save(terminal_path, terminal)
    with pytest.raises(ValueError, match=error):
        finalize(metadata["repo_root"], arm["trajectory"], arm["terminal"])
    assert not list(metadata["repo_root"].rglob("rml_finalized"))
