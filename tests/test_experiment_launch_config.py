"""Failure-oriented tests for the immutable execution launch boundary."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from molgap import kaggle_accelerator_push
from molgap.experiment_package import SIDECARS, build_experiment_source_package, verify_experiment_source_package
from molgap.experiment_launch import canonical_json
from molgap.experiment_launch_config import validate_launch_config
from molgap.experiment_preflight import check_workflow_binding
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import state_dict_sha256
from test_experiment_workflow import _arm, _candidate_pair_spec, _git, _jobs, _spec_payload


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _trajectory(spec: ExperimentSpec, arm: dict, source_commit: str, trajectory_id: str) -> dict:
    arm_id = arm["arm_id"]
    return {
        "schema": "molgap-trajectory-v1",
        "trajectory_id": trajectory_id,
        "record_mode": "prospective",
        "owner": "server",
        "track": "B",
        "family_id": arm["family"]["name"],
        "question": "Synthetic launch binding question",
        "hypothesis": {
            "hypothesis_id": "hyp-" + arm_id,
            "supporting_evidence_ids": ["evidence-support-" + arm_id],
            "alternative_explanations": ["alternative-" + arm_id],
            "related_closed_family_ids": ["closed-family-" + arm_id],
            "observed_deficiency": "Synthetic deficiency",
            "changed_mechanism": "Synthetic mechanism",
            "cheapest_falsifier": "Synthetic falsifier",
            "expected_native_cost_ref": "cost-" + arm_id,
            "decision_changed_if_positive": "Continue",
            "decision_changed_if_negative": "Stop",
        },
        "state_at_start": {
            "source_commit": source_commit,
            "source_config_identity": canonical_fingerprint(arm),
            "contract_refs": ["contract-" + arm_id],
            "reference_ids": [],
            "parent_trajectory_ids": [],
            "prior_evidence_ids": ["prior-" + arm_id],
            "role_snapshot_refs": ["role-" + arm_id],
            "budget_snapshot_ref": "budget-" + arm_id,
        },
        "actions": [{
            "action_id": "action-" + arm_id,
            "type": "train",
            "source_commit": source_commit,
            "run_ids": ["run-" + arm_id],
            "attempt_ids": ["attempt-" + arm_id],
            "evidence_refs": ["evidence-" + arm_id],
            "cost_event_ids": ["cost-" + arm_id],
        }],
        "result": {"evidence_ids": [], "evidence_refs": []},
        "decision": {
            "decision_ref": "decision-" + arm_id,
            "outcome": "ACTIVE",
            "next_allowed_actions": ["continue"],
            "reopen_conditions": ["new evidence"],
        },
    }


@pytest.fixture
def real_release_case(tmp_path):
    """A complete local release with actual package/state/trajectory bytes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src/molgap"
    source.mkdir(parents=True)
    (source / "__init__.py").write_bytes(b"")
    (source / "loader.py").write_bytes(b"VALUE = 1\n")
    recipe_paths, states, arms = {}, {}, []
    for index, arm_id in enumerate(("reference_arm", "candidate_arm")):
        recipe_path = repo / "recipes" / (arm_id + ".json")
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_bytes = canonical_json({"synthetic_recipe": arm_id}).encode("utf-8")
        recipe_path.write_bytes(recipe_bytes)
        recipe_paths[arm_id] = recipe_path.relative_to(repo).as_posix()
        state_value = {"weight": torch.tensor([float(index + 1)], dtype=torch.float32)}
        state_path = tmp_path / (arm_id + ".pt")
        torch.save(state_value, state_path)
        states[arm_id] = state_path
        arms.append(_arm("gptrans_t", "1", arm_id,
                         recipe_sha256=_sha(recipe_bytes),
                         state_sha256=state_dict_sha256(state_value)))
    _git(repo, "init")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic launch fixture")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic launch fixture")
    payload = _spec_payload(arms, platform="kaggle")
    for index, binding in enumerate(payload["prospective"]["arms"]):
        binding["trajectory_id"] = "trajectory-" + arms[index]["arm_id"]
    spec = ExperimentSpec(payload)
    package = tmp_path / "package"
    manifest = build_experiment_source_package(
        spec, repo, ["src/molgap/__init__.py", "src/molgap/loader.py", *recipe_paths.values()], package)
    input_root = tmp_path / "input"
    input_root.mkdir()
    for name in sorted(SIDECARS - {"source.tar.gz"}):
        shutil.copyfile(package / name, input_root / name)
    shutil.copyfile(package / "source.tar.gz", input_root / "source_payload.bin")
    staged_states = {}
    for arm_id, state_path in states.items():
        staged = input_root / "initial_states" / (arm_id + ".pt")
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(state_path, staged)
        staged_states[arm_id] = staged
    prospective_hashes = {}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        trajectory_id = next(item["trajectory_id"] for item in spec.to_dict()["prospective"]["arms"]
                             if item["arm_id"] == arm_id)
        path = input_root / "prospective" / arm_id / "trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(_trajectory(spec, arm, manifest["source_commit"], trajectory_id)).encode())
        prospective_hashes[arm_id] = _sha(path.read_bytes())
    jobs = [{
        "arm_id": arm["arm_id"], "device": index,
        "recipe": recipe_paths[arm["arm_id"]],
        "initial_state": "initial_states/" + arm["arm_id"] + ".pt",
        "trajectory_id": next(item["trajectory_id"] for item in spec.to_dict()["prospective"]["arms"]
                               if item["arm_id"] == arm["arm_id"]),
    } for index, arm in enumerate(spec.to_dict()["arms"])]
    launch = {
        "format": "molgap-execution-launch-v1",
        "spec_identity": spec.identity,
        "expected_package_identity": manifest["package_identity"],
        "expected_source_archive_sha256": manifest["archive_sha256"],
        "account": "synthetic-account",
        "run_reference": "synthetic-account/synthetic-run",
        "dataset_sources": ["synthetic-account/source"],
        "accelerator": "NvidiaTeslaT4",
        "device_count": 2,
        "jobs": jobs,
        "prospective_sha256": prospective_hashes,
    }
    launch_path = input_root / "experiment_launch.json"
    _write(launch_path, launch)
    metadata_path = input_root / "kernel-metadata.json"
    _write(metadata_path, {
        "id": launch["run_reference"], "dataset_sources": launch["dataset_sources"],
        "enable_gpu": True, "enable_tpu": False,
    })
    entry_path = input_root / "run.py"
    entry_path.write_text("EXPECTED_LAUNCH_SHA256 = " + repr(_sha(launch_path.read_bytes())) + "\n",
                          encoding="utf-8")
    options = {
        "expected_package_identity": manifest["package_identity"],
        "recipe_files": recipe_paths,
        "initial_states": staged_states,
        "required_modules": ["molgap.loader"],
        "entry_script": entry_path,
        "input_root": input_root,
        "launch_config": launch_path,
        "kernel_metadata": metadata_path,
    }
    return {
        "spec": spec, "repo": repo, "package": package, "manifest": manifest,
        "input_root": input_root, "launch": launch, "launch_path": launch_path,
        "metadata_path": metadata_path, "entry_path": entry_path,
        "options": options,
    }


@pytest.fixture
def launch_case(tmp_path, monkeypatch):
    spec = _candidate_pair_spec()
    root = tmp_path / "input"
    root.mkdir()
    jobs = _jobs(spec)
    initial_states = {}
    prospective = {}
    for job in jobs:
        arm_id = job["arm_id"]
        state = root / job["initial_state"]
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(("state-" + arm_id).encode())
        initial_states[arm_id] = state
        trajectory = root / "prospective" / arm_id / "trajectory.json"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_bytes(("trajectory-" + arm_id).encode())
        prospective[arm_id] = _sha(trajectory.read_bytes())

    # Keep this test focused on the launch boundary. The owning trajectory and
    # state inspectors are covered independently and run here only as trusted
    # gates, so no model, graph, or platform work is performed.
    monkeypatch.setattr(
        "molgap.experiment_execution.validate_staged_trajectory",
        lambda *_args, **_kwargs: {"state_at_start": {"source_commit": "c" * 40}},
    )
    monkeypatch.setattr(
        "molgap.v4_runtime.inspect_frozen_state_artifact",
        lambda *_args, **_kwargs: {"device": "cpu"},
    )
    launch = {
        "format": "molgap-execution-launch-v1",
        "spec_identity": spec.identity,
        "expected_package_identity": "a" * 64,
        "expected_source_archive_sha256": "b" * 64,
        "account": "synthetic-account",
        "run_reference": "synthetic-account/synthetic-run",
        "dataset_sources": ["synthetic-account/source"],
        "accelerator": "NvidiaTeslaT4",
        "device_count": 2,
        "jobs": jobs,
        "prospective_sha256": prospective,
    }
    launch_path = root / "experiment_launch.json"
    _write(launch_path, launch)
    manifest = {
        "package_identity": "a" * 64,
        "archive_sha256": "b" * 64,
        "spec_identity": spec.identity,
        "source_commit": "c" * 40,
    }
    recipes = {job["arm_id"]: job["recipe"] for job in jobs}
    return spec, root, launch_path, launch, manifest, recipes, initial_states


def _check(case, **kwargs):
    spec, root, launch_path, _launch, manifest, recipes, initial_states = case
    return validate_launch_config(
        spec, launch_path, manifest=manifest, staged_root=root,
        recipe_files=recipes, initial_states=initial_states, **kwargs,
    )


def test_strict_launch_binds_manifest_jobs_and_prospective_bytes(launch_case):
    checked = _check(launch_case)
    assert checked["format"] == "molgap-execution-launch-v1"
    assert checked["source_commit"] == "c" * 40
    assert {job["arm_id"] for job in checked["jobs"]} == {
        arm["arm_id"] for arm in launch_case[0].to_dict()["arms"]
    }


@pytest.mark.parametrize("field", [
    "expected_package_identity", "expected_source_archive_sha256", "jobs",
    "prospective_sha256",
])
def test_partial_marked_launch_fails_closed(launch_case, field):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    changed = copy.deepcopy(launch)
    changed.pop(field)
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="missing or unknown fields"):
        _check(launch_case)


def test_launch_requires_verified_package_in_addition_to_expected_identity(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    with pytest.raises(ValueError, match="verified package manifest or package directory"):
        validate_launch_config(spec, launch_path, staged_root=root,
                               expected_package_identity=manifest["package_identity"])


def test_launch_rejects_manifest_archive_rebind(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    bad = dict(manifest, archive_sha256="d" * 64)
    with pytest.raises(ValueError, match="archive hash"):
        validate_launch_config(spec, launch_path, manifest=bad, staged_root=root,
            recipe_files=recipes, initial_states=initial_states)


def test_launch_rejects_job_recipe_or_initial_path_rebind(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    changed = copy.deepcopy(launch)
    changed["jobs"][0]["recipe"] = "recipes/other.json"
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="recipe path"):
        _check(launch_case)

    changed = copy.deepcopy(launch)
    changed["jobs"][0]["initial_state"] = "initial_states/other.pt"
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="initialization input"):
        _check(launch_case)


def test_launch_rejects_job_trajectory_rebind(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    changed = copy.deepcopy(launch)
    changed["jobs"][0]["trajectory_id"] = "trajectory-from-another-arm"
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="trajectory mismatch"):
        _check(launch_case)


def test_launch_rejects_prospective_source_identity_rebind(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    bad = dict(manifest, source_commit="d" * 40)
    with pytest.raises(ValueError, match="Prospective source identity"):
        validate_launch_config(spec, launch_path, manifest=bad, staged_root=root,
            recipe_files=recipes, initial_states=initial_states)


def test_resume_is_all_arm_manifest_mapping_and_hash_bound(launch_case, monkeypatch):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    resume_calls = []

    def _resume_gate(path, checked_spec, *, arm_id, context, trajectory):
        # This test exercises launch mapping/schema and the identity handoff to
        # the resume owner.  Full bundle transport and family validation have
        # their own integration coverage; keep the fixture intentionally small.
        resume_calls.append({
            "path": Path(path),
            "spec": checked_spec,
            "arm_id": arm_id,
            "context": context,
            "trajectory": Path(trajectory),
        })
        return {"status": "RESUME_VERIFIED"}

    monkeypatch.setattr("molgap.experiment_resume.validate_resume_bundle", _resume_gate)
    changed = copy.deepcopy(launch)
    resume = {}
    for job in changed["jobs"]:
        arm_id = job["arm_id"]
        path = root / "resume" / arm_id / "resume_manifest.json"
        _write(path, {"arm_id": arm_id, "status": "resume-ready"})
        resume[arm_id] = {"manifest": path.relative_to(root).as_posix(),
                          "sha256": _sha(path.read_bytes())}
    changed["resume"] = resume
    _write(launch_path, changed)
    assert _check(launch_case)["resume"] == resume
    assert {item["arm_id"] for item in resume_calls} == set(resume)
    assert len(resume_calls) == len(resume)
    for item in resume_calls:
        arm_id = item["arm_id"]
        context = item["context"]
        assert item["path"] == root / resume[arm_id]["manifest"]
        assert item["spec"] is spec
        assert context.spec_identity == spec.identity
        assert context.arm_id == arm_id
        assert context.package_identity == manifest["package_identity"]
        assert context.source_commit == manifest["source_commit"]
        assert context.source_archive_sha256 == manifest["archive_sha256"]
        assert context.account == launch["account"]
        assert context.run_reference == launch["run_reference"]
        assert item["trajectory"] == root / "prospective" / arm_id / "trajectory.json"

    changed["resume"].pop(next(iter(resume)))
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="every Spec arm"):
        _check(launch_case)


def test_unmarked_legacy_launch_is_left_to_compatibility_gate(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    changed = {"spec_identity": spec.identity, "run_reference": "synthetic-account/synthetic-run",
               "account": "synthetic-account", "dataset_sources": ["synthetic-account/source"],
               "accelerator": "NvidiaTeslaT4", "device_count": 2,
               "prospective_sha256": launch["prospective_sha256"]}
    _write(launch_path, changed)
    with pytest.raises(ValueError, match="unsupported or missing format"):
        validate_launch_config(spec, launch_path, manifest=manifest, staged_root=root)


def test_workflow_binding_rejects_missing_format_instead_of_legacy_fallback(launch_case):
    spec, root, launch_path, launch, manifest, recipes, initial_states = launch_case
    unmarked = {key: launch[key] for key in (
        "spec_identity", "run_reference", "account", "dataset_sources",
        "accelerator", "device_count", "prospective_sha256",
    )}
    _write(launch_path, unmarked)
    metadata = root / "kernel-metadata.json"
    _write(metadata, {
        "id": launch["run_reference"], "dataset_sources": launch["dataset_sources"],
        "enable_gpu": True, "enable_tpu": False,
    })
    entry = root / "run.py"
    entry.write_text("EXPECTED_LAUNCH_SHA256 = " + repr(_sha(launch_path.read_bytes())) + "\n",
                     encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported or missing format"):
        check_workflow_binding(
            spec, launch_config=launch_path, kernel_metadata=metadata,
            entry_script=entry, manifest=manifest, staged_root=root,
            recipe_files=recipes, initial_states=initial_states,
        )


def test_real_release_check_and_kaggle_report_reject_launch_rebindings(real_release_case):
    case = real_release_case
    report = check_release_inputs(case["spec"], case["package"], **case["options"])
    assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED", report
    assert report["errors"] == []
    report_path = case["input_root"] / "release.json"
    report_path.write_bytes(canonical_json(report).encode("utf-8"))
    assert kaggle_accelerator_push._verify_release_report(
        report_path, case["entry_path"]
    )["package_identity"] == case["manifest"]["package_identity"]

    original = copy.deepcopy(case["launch"])
    mutations = {
        "wrong_package": lambda value: value.update(expected_package_identity="a" * 64),
        "wrong_archive": lambda value: value.update(expected_source_archive_sha256="b" * 64),
        "duplicate_device": lambda value: value["jobs"][1].update(device=value["jobs"][0]["device"]),
        "missing_jobs": lambda value: value.pop("jobs"),
        "missing_format": lambda value: value.pop("format"),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        _write(case["launch_path"], changed)
        observed = check_release_inputs(case["spec"], case["package"], **case["options"])
        assert observed["status"] == "RELEASE_INPUTS_FAILED", name
        assert observed["errors"], name

    # The old verified receipt cannot be replayed after the launch bytes move.
    _write(case["launch_path"], {**original, "expected_source_archive_sha256": "c" * 64})
    with pytest.raises(ValueError, match="changed after verification"):
        kaggle_accelerator_push._verify_release_report(report_path, case["entry_path"])
