"""Execution retention tests; scientific family manifests remain untouched."""
from __future__ import annotations

import json

import pytest

from molgap.experiment_allocation import AllocationLedger
from molgap.experiment_retention import (
    LEDGER_FORMAT,
    RETENTION_FILENAME,
    seal_execution_retention,
    validate_execution_retention,
)
from molgap.experiment_spec import ExperimentSpec
from test_experiment_workflow import _candidate_pair_spec


PACKAGE_ID = "a" * 64
ARCHIVE_ID = "b" * 64
SOURCE_COMMIT = "c" * 40


@pytest.fixture
def retention_case(tmp_path):
    spec = _candidate_pair_spec()
    root = tmp_path / "execution"
    root.mkdir()
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    arm_roots = {arm_id: root / arm_id for arm_id in arm_ids}
    clock = [0.0]
    prior = {"format": LEDGER_FORMAT, "spec_identity": spec.identity, "status": "running"}
    ledger = AllocationLedger(
        spec_identity=spec.identity,
        hardware=["Tesla T4", "Tesla T4"],
        assignments={arm_ids[0]: 0, arm_ids[1]: 1},
        started=0.0,
        prior_segments=[prior],
        clock=lambda: clock[0],
    )
    clock[0] = 12.5
    ledger.write(root, "complete", arm_roots=list(arm_roots.values()))
    (root / "execution_report.json").write_text(
        json.dumps({"format": "molgap-kaggle-two-phase-pair-v2", "status": "complete"}),
        encoding="utf-8",
    )
    return {
        "spec": spec,
        "root": root,
        "arm_roots": arm_roots,
        "arm_ids": arm_ids,
        "ledger": ledger,
    }


@pytest.fixture
def single_arm_retention_case(tmp_path):
    payload = _candidate_pair_spec().to_dict()
    payload["arms"] = payload["arms"][:1]
    payload["prospective"]["arms"] = payload["prospective"]["arms"][:1]
    spec = ExperimentSpec(payload)
    root = tmp_path / "single-arm-execution"
    root.mkdir()
    arm_id = spec.to_dict()["arms"][0]["arm_id"]
    arm_root = root / arm_id
    clock = [0.0]
    ledger = AllocationLedger(
        spec_identity=spec.identity,
        hardware=["Tesla T4", "Tesla T4"],
        assignments={arm_id: 0},
        started=0.0,
        clock=lambda: clock[0],
    )
    clock[0] = 12.5
    ledger.write(root, "complete", arm_roots=[arm_root])
    return {"spec": spec, "root": root, "arm_roots": {arm_id: arm_root},
            "arm_ids": [arm_id], "ledger": ledger}


def _seal(case):
    return seal_execution_retention(
        case["root"], case["spec"], package_identity=PACKAGE_ID,
        source_commit=SOURCE_COMMIT, source_archive_sha256=ARCHIVE_ID,
        arm_roots=case["arm_roots"],
    )


def _validate(case):
    return validate_execution_retention(
        case["root"], case["spec"], package_identity=PACKAGE_ID,
        source_commit=SOURCE_COMMIT, source_archive_sha256=ARCHIVE_ID,
    )


def test_seal_and_validate_binds_root_and_every_arm_ledger(retention_case):
    case = retention_case
    sealed = _seal(case)
    assert sealed["status"] == "EXECUTION_RETENTION_SEALED"
    assert sealed["manifest_path"].name == RETENTION_FILENAME
    assert sealed["manifest"]["execution_report"]["path"] == "execution_report.json"
    checked = _validate(case)
    assert checked["status"] == "EXECUTION_RETENTION_VERIFIED"
    assert set(checked["ledgers"]["arms"]) == set(case["arm_ids"])
    assert checked["ledgers"]["root"]["prior_unobserved_intervals"] is True


def test_idle_allocated_device_is_retained_without_claiming_assignment(single_arm_retention_case):
    case = single_arm_retention_case
    sealed = _seal(case)
    ledger = sealed["ledgers"]["root"]
    assert ledger["devices"][1]["arm_id"] is None
    assert ledger["unassigned_device_seconds"] == ledger["wall_seconds"]
    assert ledger["allocated_device_seconds"] == 2 * ledger["wall_seconds"]


def test_root_allocation_matches_spec_device_count_and_arm_assignments(retention_case):
    case = retention_case
    root_ledger = case["root"] / "allocation_ledger.json"
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["allocation_device_count"] = 1
    changed["devices"] = changed["devices"][:1]
    changed["allocated_device_seconds"] = changed["wall_seconds"]
    changed["unassigned_device_seconds"] = 0
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from Spec platform.device_count"):
        _seal(case)

    case["ledger"].write(case["root"], "complete", arm_roots=list(case["arm_roots"].values()))
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["devices"][0]["arm_id"] = None
    changed["unassigned_device_seconds"] = changed["wall_seconds"]
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="assign every Spec arm"):
        _seal(case)


def test_root_hardware_must_match_spec_accelerator(retention_case):
    case = retention_case
    root_ledger = case["root"] / "allocation_ledger.json"
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["devices"][0]["hardware"] = "NVIDIA A100"
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hardware differs from Spec platform accelerator"):
        _seal(case)


def test_ledger_timestamps_require_utc_and_observed_after_started(retention_case):
    case = retention_case
    root_ledger = case["root"] / "allocation_ledger.json"
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["started_at"] = "2026-01-01T00:00:00"
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit UTC"):
        _seal(case)

    case["ledger"].write(case["root"], "complete", arm_roots=list(case["arm_roots"].values()))
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["observed_at"] = "2000-01-01T00:00:00+00:00"
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="precedes started_at"):
        _seal(case)


def test_corrupt_arm_ledger_fails_after_sealing(retention_case):
    case = retention_case
    _seal(case)
    path = case["root"] / case["arm_ids"][0] / "allocation_ledger.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["devices"][1]["arm_id"] = case["arm_ids"][1]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after sealing"):
        _validate(case)


def test_invalid_status_or_prior_identity_cannot_be_sealed(retention_case):
    case = retention_case
    root_ledger = case["root"] / "allocation_ledger.json"
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["status"] = "unknown"
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid status"):
        _seal(case)

    # Recreate a valid ledger, then make a retained historical segment cross
    # the current Spec. The current unknown interval bit cannot legitimize it.
    case["ledger"].write(case["root"], "complete", arm_roots=list(case["arm_roots"].values()))
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    changed["prior_segments"][0]["spec_identity"] = "d" * 64
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses the execution Spec identity"):
        _seal(case)

    # A full historical segment may itself retain prior segments.  That nested
    # identity must be checked instead of being skipped as an opaque segment.
    case["ledger"].write(case["root"], "complete", arm_roots=list(case["arm_roots"].values()))
    changed = json.loads(root_ledger.read_text(encoding="utf-8"))
    nested = dict(changed)
    nested["prior_segments"] = [{
        "format": LEDGER_FORMAT,
        "spec_identity": "d" * 64,
        "status": "complete",
    }]
    nested["prior_unobserved_intervals"] = True
    changed["prior_segments"] = [nested]
    changed["prior_unobserved_intervals"] = True
    root_ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses the execution Spec identity"):
        _seal(case)


def test_retention_allowlist_does_not_attach_arbitrary_output(retention_case):
    case = retention_case
    (case["root"] / "unrelated-debug-dump.bin").write_bytes(b"ignored")
    sealed = _seal(case)
    manifest = sealed["manifest"]
    assert "unrelated-debug-dump.bin" not in json.dumps(manifest)
    assert set(manifest["allocation_ledger"]) == {"root", "arms"}
