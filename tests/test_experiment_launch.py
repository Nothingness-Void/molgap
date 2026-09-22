"""Local synthetic receipts only. Authored for later execution, no live adapters."""
import copy
import json
from pathlib import Path
import socket
import subprocess

import pytest

import molgap.experiment_launch as launch
from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_package import package, repo  # Synthetic local Git/package fixtures.
from test_experiment_spec import payload


def unknown(reason="not_reported"):
    return {"value": None, "missing_reason": reason}


def known(value):
    return {"value": value, "missing_reason": None}


@pytest.fixture
def context(package, payload):
    spec = ExperimentSpec(payload)
    pin = verify_experiment_source_package(package)["package_identity"]
    return spec, package, pin


def build(context, response=None):
    spec, directory, pin = context
    return launch.build_launch_receipt(spec, directory, expected_package_identity=pin,
                                       response_json=None if response is None else launch.canonical_json(response))


def envelope(context, outcome="accepted"):
    return {
        "format": launch.RESPONSE_FORMAT, "version": 1, "mode": "observed",
        "outcome": outcome, "conflict_kind": None, "binding": build(context)["binding"],
        "canonical_platform_reference": known("synthetic-owner/workload"),
        "platform_version": unknown(), "physical_runs": unknown(),
        "timestamp": unknown(), "monitor_paths": unknown(),
    }


def persist(context, receipt, directory):
    spec, package_dir, pin = context
    return launch.write_launch_receipt(launch.canonical_json(receipt), directory, spec,
                                       package_dir, expected_package_identity=pin)


def test_exact_binding_and_source_unchanged(context, tmp_path):
    spec, directory, pin = context
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    receipt = build(context)
    binding = receipt["binding"]
    manifest = verify_experiment_source_package(directory)
    assert binding["package_identity"] == pin
    assert binding["spec_identity"] == spec.identity
    assert binding["spec_sha256"] == manifest["spec_sha256"]
    assert binding["source_commit"] == manifest["source_commit"]
    assert binding["source_archive_sha256"] == manifest["archive_sha256"]
    assert binding["arms"] == [{"arm_id": a["arm_id"], "arm_identity": canonical_fingerprint(a)}
                               for a in spec.to_dict()["arms"]]
    assert receipt["submission_state"] == "NOT_SUBMITTED"
    assert receipt["submitter_status"] == "SUBMIT_UNIMPLEMENTED"
    path = persist(context, receipt, tmp_path)
    assert launch.read_launch_receipt(path, spec, directory, expected_package_identity=pin) == receipt
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("outcome,state", [
    ("accepted", "ACCEPTED"), ("queued", "PENDING_RECONCILIATION"),
    ("pending", "PENDING_RECONCILIATION"), ("client_unknown_after_submit", "PENDING_RECONCILIATION"),
    ("existing_found", "RECONCILED_EXISTING"), ("rejected_permission", "REJECTED"),
    ("unknown", "UNKNOWN"),
])
def test_explicit_outcomes(context, outcome, state):
    response = envelope(context, outcome)
    spec, directory, pin = context
    receipt = launch.reconcile_platform_response(spec, directory, launch.canonical_json(response),
                                                 expected_package_identity=pin)
    assert receipt["submission_state"] == state
    assert receipt == build(context, response)
    assert receipt["cost"] == unknown("not_observed")


@pytest.mark.parametrize("kind", ["name_conflict", "existing_version", "permission", "other"])
def test_409_never_retries_or_renames(context, kind):
    response = envelope(context, "rejected_409")
    response["conflict_kind"] = kind
    receipt = build(context, response)
    assert receipt["submission_state"] == "REJECTED"
    assert receipt["reconciliation_result"] == "REJECTED_409_" + kind.upper()
    assert receipt["binding"] == response["binding"]
    assert receipt["next_action"] == "RESOLVE_WITHOUT_AUTOMATIC_RETRY"


@pytest.mark.parametrize("outcome", ["accepted", "existing_found"])
@pytest.mark.parametrize("field", ["spec_identity", "spec_sha256", "package_identity", "source_commit",
                                   "source_archive_sha256", "arms", "logical_run_id", "requested_logical_platform"])
def test_identity_mismatch_is_conflict(context, outcome, field):
    response = envelope(context, outcome)
    binding = response["binding"]
    if field == "arms":
        binding[field].reverse()
    elif field == "logical_run_id":
        binding[field] = "other-run"
    elif field == "requested_logical_platform":
        binding[field] = "kaggle"
    else:
        binding[field] = "0" * len(binding[field])
    receipt = build(context, response)
    assert receipt["submission_state"] == "REJECTED"
    assert receipt["reconciliation_result"] == "REJECTED_CONFLICT"
    assert receipt["canonical_platform_reference"]["value"] is None


def test_arm_identity_mismatch(context):
    response = envelope(context, "existing_found")
    response["binding"]["arms"][0]["arm_identity"] = "0" * 64
    assert build(context, response)["reconciliation_result"] == "REJECTED_CONFLICT"


def test_two_arms_do_not_imply_two_external_runs(context):
    response = envelope(context)
    assert len(response["binding"]["arms"]) == 2
    assert build(context, response)["physical_runs"] == unknown()
    ids = [a["arm_id"] for a in response["binding"]["arms"]]
    response["physical_runs"] = known([{
        "run_identity": "one-observed-run", "canonical_reference": "owner/workload",
        "platform_version": known("7"), "arm_ids": ids,
    }])
    assert len(build(context, response)["physical_runs"]["value"]) == 1
    response["physical_runs"]["value"] = [
        {"run_identity": f"observed-{i}", "canonical_reference": "owner/workload",
         "platform_version": unknown(), "arm_ids": [arm_id]}
        for i, arm_id in enumerate(ids)
    ]
    assert len(build(context, response)["physical_runs"]["value"]) == 2


@pytest.mark.parametrize("change", ["duplicate_run", "duplicate_arm", "unknown_arm", "partial", "order"])
def test_bad_physical_mapping(context, change):
    response = envelope(context)
    ids = [a["arm_id"] for a in response["binding"]["arms"]]
    runs = [{"run_identity": "run", "canonical_reference": "owner/workload",
             "platform_version": unknown(), "arm_ids": ids.copy()}]
    if change == "duplicate_run":
        runs.append(copy.deepcopy(runs[0]))
    elif change == "duplicate_arm":
        runs[0]["arm_ids"].append(ids[0])
    elif change == "unknown_arm":
        runs[0]["arm_ids"].append("other")
    elif change == "partial":
        runs[0]["arm_ids"].pop()
    else:
        runs[0]["arm_ids"].reverse()
    response["physical_runs"] = known(runs)
    with pytest.raises(ValueError):
        build(context, response)


def test_unknowns_and_lost_response(context):
    response = envelope(context, "client_unknown_after_submit")
    response["canonical_platform_reference"] = unknown("response_lost")
    receipt = build(context, response)
    assert receipt["submission_state"] == "PENDING_RECONCILIATION"
    for key in ("physical_runs", "timestamp", "platform_version", "cost", "canonical_platform_reference"):
        assert receipt[key]["value"] is None
        assert receipt[key]["missing_reason"]


def test_known_observations_are_preserved(context):
    response = envelope(context)
    response["timestamp"] = known("2026-09-22T12:34:56+00:00")
    response["monitor_paths"] = known(["logs/train.stdout.log", "status.json"])
    response["platform_version"] = known("7")
    receipt = build(context, response)
    for key in ("timestamp", "monitor_paths", "platform_version"):
        assert receipt[key] == response[key]
    assert receipt["cost"] == unknown("not_observed")


@pytest.mark.parametrize("outcome", ["accepted", "existing_found", "queued"])
def test_dry_run_never_claims_real_submission(context, outcome):
    response = envelope(context, outcome)
    response["mode"] = "dry_run"
    receipt = build(context, response)
    assert receipt["submission_state"] == "NOT_SUBMITTED"
    assert receipt["reconciliation_result"] == "DRY_RUN_ONLY"
    assert receipt["canonical_platform_reference"]["value"] is None


@pytest.mark.parametrize("mutation", ["version", "bool_version", "format", "outcome", "mode", "409",
                                     "reason", "known_reason", "timestamp", "path", "duplicate_paths",
                                     "existing_unknown", "duplicate_arm"])
def test_strict_envelope(context, mutation):
    response = envelope(context)
    if mutation in {"format", "outcome", "mode"}:
        response[mutation] = "unrecognized"
    elif mutation == "version":
        response["version"] = 2
    elif mutation == "bool_version":
        response["version"] = True
    elif mutation == "409":
        response["outcome"] = "rejected_409"
    elif mutation == "reason":
        response["timestamp"]["missing_reason"] = None
    elif mutation == "known_reason":
        response["canonical_platform_reference"]["missing_reason"] = "not_reported"
    elif mutation == "timestamp":
        response["timestamp"] = known("2026-09-22")
    elif mutation == "path":
        response["monitor_paths"] = known(["../secret"])
    elif mutation == "duplicate_paths":
        response["monitor_paths"] = known(["logs/out", "logs/OUT"])
    elif mutation == "existing_unknown":
        response["outcome"] = "existing_found"
        response["canonical_platform_reference"] = unknown()
    else:
        response["binding"]["arms"].append(copy.deepcopy(response["binding"]["arms"][0]))
    with pytest.raises(ValueError):
        build(context, response)


@pytest.mark.parametrize("field", ["callback", "dynamic_import", "eval", "token", "ready_for_desktop",
                                 "rml_closed", "replay_ready", "preflight_pass", "cost"])
def test_forbidden_extra_fields(context, field):
    response = envelope(context)
    response[field] = "not-allowed"
    with pytest.raises(ValueError):
        build(context, response)


@pytest.mark.parametrize("kind", ["pretty", "duplicate", "newline", "nan", "dict", "callback"])
def test_noncanonical_response(context, kind):
    spec, directory, pin = context
    response = envelope(context)
    raw = launch.canonical_json(response)
    raw = {"pretty": json.dumps(response, indent=2), "duplicate": '{"version":1,' + raw[1:],
           "newline": raw + "\n", "nan": raw.replace('"version":1', '"version":NaN'),
           "dict": response, "callback": lambda: response}[kind]
    with pytest.raises((ValueError, TypeError)):
        launch.reconcile_platform_response(spec, directory, raw, expected_package_identity=pin)


def test_idempotent_and_retry_conflicts(context, tmp_path):
    receipt = build(context)
    path = persist(context, receipt, tmp_path)
    before = path.stat().st_mtime_ns
    assert persist(context, receipt, tmp_path) == path
    assert path.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="conflict"):
        persist(context, build(context, envelope(context)), tmp_path)
    assert path.read_bytes() == launch.canonical_json(receipt).encode()
    assert not list(tmp_path.glob(".launch-*"))


@pytest.mark.parametrize("change", ["state", "cost", "authority", "identity", "newline", "duplicate", "empty"])
def test_receipt_tamper_and_overwrite(context, tmp_path, change):
    receipt = build(context)
    path = persist(context, receipt, tmp_path)
    altered = copy.deepcopy(receipt)
    if change == "state":
        altered["submission_state"] = "ACCEPTED"
    elif change == "cost":
        altered["cost"] = known(0)
    elif change == "authority":
        altered["replay_ready"] = True
    elif change == "identity":
        altered["receipt_identity"] = "0" * 64
    raw = launch.canonical_json(altered)
    if change == "newline":
        raw += "\n"
    elif change == "duplicate":
        raw = '{"version":1,' + raw[1:]
    elif change == "empty":
        raw = ""
    path.write_bytes(raw.encode())
    spec, directory, pin = context
    with pytest.raises(ValueError):
        launch.read_launch_receipt(path, spec, directory, expected_package_identity=pin)
    with pytest.raises(ValueError):
        persist(context, receipt, tmp_path)


@pytest.mark.parametrize("sidecar", ["source.tar.gz", "package_manifest.json", "experiment_spec.json",
                                    "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"])
def test_package_tamper(context, sidecar):
    path = context[1] / sidecar
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        build(context)


def test_pin_spec_and_order(context, payload):
    spec, directory, pin = context
    with pytest.raises(ValueError):
        build((spec, directory, "0" * 64))
    payload["arms"].reverse()
    with pytest.raises(ValueError):
        build((ExperimentSpec(payload), directory, pin))
    class Derived(ExperimentSpec):
        pass
    with pytest.raises(TypeError):
        build((Derived(payload), directory, pin))
    object.__setattr__(spec, "_canonical_json", json.dumps(spec.to_dict(), indent=2))
    with pytest.raises(ValueError):
        build(context)


@pytest.mark.parametrize("kind", ["relative", "traversal", "missing", "package", "rml", "stream", "device"])
def test_output_path_safety(context, tmp_path, kind):
    receipt = build(context)
    paths = {"relative": Path("receipts"), "traversal": tmp_path / "x" / ".." / "receipts",
             "missing": tmp_path / "missing", "package": context[1],
             "rml": tmp_path / "research_memory", "stream": tmp_path / "out:stream",
             "device": tmp_path / "NUL"}
    if kind == "rml":
        paths[kind].mkdir()
    with pytest.raises(ValueError):
        persist(context, receipt, paths[kind])


def test_symlink_output(context, tmp_path):
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit symlinks")
    with pytest.raises(ValueError):
        persist(context, build(context), alias)


def test_atomic_publish_failure(context, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic publication failure")
    monkeypatch.setattr(launch.os, "link", fail)
    with pytest.raises(OSError):
        persist(context, build(context), tmp_path)
    assert not list(tmp_path.glob(".launch-*"))
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("same", [True, False])
def test_atomic_publish_race(context, tmp_path, monkeypatch, same):
    receipt = build(context)
    expected = launch.canonical_json(receipt).encode()
    winner = expected if same else b"conflicting-winner"
    def competing_writer(source, target):
        target.write_bytes(winner)
        raise FileExistsError("synthetic competing publisher")
    monkeypatch.setattr(launch.os, "link", competing_writer)
    if same:
        path = persist(context, receipt, tmp_path)
        assert path.read_bytes() == expected
    else:
        with pytest.raises(ValueError, match="conflict"):
            persist(context, receipt, tmp_path)
    assert (tmp_path / (receipt["launch_identity"] + ".json")).read_bytes() == winner
    assert not list(tmp_path.glob(".launch-*"))


def test_writer_rejects_noncanonical_or_forged_input(context, tmp_path):
    spec, directory, pin = context
    receipt = build(context)
    for raw in (json.dumps(receipt, indent=2), launch.canonical_json(receipt) + "\n",
                '{"version":1,' + launch.canonical_json(receipt)[1:]):
        with pytest.raises(ValueError):
            launch.write_launch_receipt(raw, tmp_path, spec, directory, expected_package_identity=pin)
    receipt["submission_state"] = "ACCEPTED"
    with pytest.raises(ValueError):
        persist(context, receipt, tmp_path)


def test_no_process_network_or_authority(context, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected external call")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    receipt = build(context, envelope(context))
    persist(context, receipt, tmp_path)
    forbidden_keys = {"ready", "ready_for_desktop", "rml_closed", "replay_ready", "scientific_acceptance"}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden_keys & value.keys()
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(receipt)
