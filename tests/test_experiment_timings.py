"""Synthetic CPU timing evidence only; no SDK, credentials, or real runs."""
import json
from pathlib import Path

import pytest

from molgap import experiment_timings as owner


IDENTITY = {"spec_identity": "a" * 64, "package_identity": "b" * 64,
            "source_commit": "c" * 40}
STAMP = "2026-10-17T00:00:00+00:00"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def observation(run, index=1, **updates):
    value = {"format": "molgap-local-preparation-observation-v1", "observed_at": STAMP,
             "status": "PREPARED_FOR_PLATFORM", **IDENTITY,
             "reused_preparation": False, "local_call_seconds": 12.5, **updates}
    return write(run / "local_observations" / f"preparation-{index:032x}.json", value)


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "run"
    write(path / "inputs/input_binding.json", IDENTITY)
    write(path / "prepared/package/package_manifest.json", IDENTITY)
    write(path / "prepared/workflow_report.json", {"status": "PREPARED_FOR_PLATFORM", **IDENTITY})
    return path


def samples(report, metric):
    return [item for item in report["samples"] if item["metric"] == metric]


def test_missing_observations_phases_and_costs_are_null_not_zero(run):
    report = owner.report_experiment_timings(run)
    assert all(item["seconds"] is None and item["missing_reason"] for item in report["samples"])
    assert all(item["sample_count"] == 0 and item["p50_seconds"] is None
               and item["p95_seconds"] is None for item in report["summaries"])
    assert report["compute_cost"]["value"] is None
    assert set(report["targets"].values()) == {"NOT_ESTABLISHED"}
    assert report["scheduler_observation"]["queue_seconds"] is None


def test_phase_accumulators_never_measure_offline_wait_receipt_latency_or_queue(run):
    observation(run)
    timings = {"NEW": 1, "SOURCE_SUBMITTING": 2, "SOURCE_PUBLISHED": 3,
               "SOURCE_PENDING": 4, "KERNEL_SUBMITTING": 5, "KERNEL_ACCEPTED": 6, "BOUND": 7}
    write(run / "operations/submission_state.json", {
        "binding": {"package_identity": IDENTITY["package_identity"]}, "phase": "BOUND",
        "timings": timings, "observed_at": "2026-10-18T00:00:00Z"})
    write(run / "operations/scheduler_observation.json", {"status": "RUNNING", "observed_at": STAMP})
    write(run / "operations/launch_receipt.json", {"binding": IDENTITY,
        "submission_state": "ACCEPTED", "timestamp": {"value": "2026-10-19T00:00:00Z", "missing_reason": None}})
    report = owner.report_experiment_timings(run)
    phases = {item["phase"]: item for item in report["samples"] if item.get("phase")}
    for phase, seconds in timings.items():
        assert phases[phase]["seconds"] == seconds
        assert phases[phase]["sample_kind"] == "active_phase_accumulator"
        assert "excludes" in phases[phase]["qualification"]
    for metric in ("source_ready_wait", "receipt_end_to_end", "training_queue", "upload_transfer"):
        assert samples(report, metric)[0]["seconds"] is None
    assert [s["phase"] for s in samples(report, "source_ready_checks_active")] == ["SOURCE_PUBLISHED", "SOURCE_PENDING"]
    assert report["receipt_observation"]["present"]
    assert report["scheduler_observation"]["status"] == "RUNNING"
    assert "total_seconds" not in report
    assert set(report["targets"].values()) == {"NOT_ESTABLISHED"}


def test_local_raw_samples_quantiles_and_reuse_groups_are_distinct(run):
    observation(run, 1, local_call_seconds=10)
    observation(run, 2, local_call_seconds=30)
    observation(run, 3, local_call_seconds=2, reused_preparation=True)
    observation(run, 4, local_call_seconds=80, package_identity="d" * 64)
    report = owner.report_experiment_timings(run)
    raw = samples(report, "local_preparation")
    assert [item["seconds"] for item in raw] == [10, 30, 2, 80]
    assert all(item["qualification"] for item in raw)
    groups = [g for g in report["summaries"] if g["metric"] == "local_preparation"]
    assert len(groups) == 3
    ordinary = next(g for g in groups if g["sample_count"] == 2)
    assert ordinary["p50_seconds"] == 20
    assert ordinary["p95_seconds"] == pytest.approx(29)
    assert ordinary["metadata_matching_sample_count"] == 2
    assert "insufficient" in ordinary["interpretation"]
    mismatch = next(s for s in raw if s["seconds"] == 80)
    assert not mismatch["identity_matches_retained_metadata"]
    assert "package_identity_binding_mismatch" in mismatch["qualification_reasons"]
    assert "metadata strings only" in " ".join(report["limitations"])


@pytest.mark.parametrize("field", ["source_commit", "package_identity"])
def test_nullable_legacy_identity_retains_duration_without_metadata_matching(run, field):
    observation(run, **{field: None})
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] == 12.5 and item["missing_reason"] is None
    assert not item["identity_matches_retained_metadata"]
    assert field + "_unavailable" in item["qualification_reasons"]


def test_legacy_source_field_absent_is_not_invalid_or_substituted_with_current_head(run):
    path = observation(run)
    value = json.loads(path.read_text())
    del value["source_commit"]
    write(path, value)
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] == 12.5 and item["identity"]["source_commit"] is None
    assert not item["identity_matches_retained_metadata"]


def test_full_observation_without_retained_bindings_cannot_match_metadata(tmp_path):
    observation(tmp_path)
    item = samples(owner.report_experiment_timings(tmp_path), "local_preparation")[0]
    assert item["seconds"] == 12.5
    assert not item["identity_matches_retained_metadata"]
    assert "source_commit_retained_binding_unavailable" in item["qualification_reasons"]


def test_conflicting_retained_metadata_is_not_silently_accepted(run):
    observation(run)
    write(run / "prepared/package/package_manifest.json", {**IDENTITY, "source_commit": "d" * 40})
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] == 12.5 and not item["identity_matches_retained_metadata"]
    assert "source_commit_binding_mismatch" in item["qualification_reasons"]


@pytest.mark.parametrize("duration", [-1, True, "2", None, {"seconds": 1}, 10 ** 400])
def test_invalid_local_and_phase_durations_have_missing_reason(run, duration):
    observation(run, local_call_seconds=duration)
    write(run / "operations/submission_state.json", {"timings": {"KERNEL_SUBMITTING": duration}})
    report = owner.report_experiment_timings(run)
    for metric in ("local_preparation", "kernel_post_active"):
        item = samples(report, metric)[0]
        assert item["seconds"] is None and item["missing_reason"] == "invalid_duration"


@pytest.mark.parametrize("raw", ['{"timings":{"NEW":1,"NEW":2}}',
                                '{"timings":{"NEW":NaN}}', '{"timings":{"NEW":Infinity}}',
                                '{"timings":{"NEW":1e999}}', '{"timings":', '[]'])
def test_corrupt_journal_is_reported_without_derived_phase_durations(run, raw):
    path = run / "operations/submission_state.json"
    path.parent.mkdir()
    path.write_text(raw, encoding="utf-8")
    report = owner.report_experiment_timings(run)
    assert samples(report, "kernel_post_active")[0]["seconds"] is None
    assert any(d["reason"] == "invalid_json_or_object" for d in report["diagnostics"])
    json.dumps(report, allow_nan=False)


def test_unknown_and_corrupt_phases_do_not_contaminate_valid_phase(run):
    write(run / "operations/submission_state.json", {"binding": [], "phase": "SUBMIT_UNKNOWN",
        "error_type": "TimeoutError", "timings": {"FUTURE_PHASE": 9, "NEW": -1, "SOURCE_SUBMITTING": 2}})
    report = owner.report_experiment_timings(run)
    unknown = samples(report, "unclassified_phase_active")[0]
    assert unknown["seconds"] == 9 and unknown["phase"] == "FUTURE_PHASE"
    assert not unknown["identity_matches_retained_metadata"]
    assert samples(report, "submission_prechecks_active")[0]["seconds"] is None
    assert samples(report, "source_publication_request_active")[0]["seconds"] == 2
    assert report["failure_observations"][0]["reason"] == "TimeoutError"


@pytest.mark.parametrize("timings", [None, [], "missing"])
def test_missing_or_invalid_phase_container(run, timings):
    write(run / "operations/submission_state.json", {"timings": timings})
    item = samples(owner.report_experiment_timings(run), "kernel_post_active")[0]
    assert item["seconds"] is None and item["missing_reason"] == "missing_or_invalid_timings"


@pytest.mark.parametrize("phase", [None, [], {}, 123])
def test_malformed_current_phase_does_not_discard_valid_raw_timings(run, phase):
    write(run / "operations/submission_state.json", {"phase": phase, "timings": {"NEW": 3}})
    report = owner.report_experiment_timings(run)
    assert report["journal_phase"] is None
    assert samples(report, "submission_prechecks_active")[0]["seconds"] == 3


def test_explicit_measured_zero_is_distinct_from_missing_duration(run):
    observation(run, local_call_seconds=0)
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] == 0 and item["missing_reason"] is None


def test_success_only_observations_never_turn_failures_into_zero_latency(run):
    observation(run, status="BLOCKED")
    write(run / "prepared/workflow_report.json", {"status": "ERROR_BEFORE_PUBLICATION", "reason": "secret arbitrary message"})
    report = owner.report_experiment_timings(run)
    item = samples(report, "local_preparation")[0]
    assert item["seconds"] is None and item["missing_reason"] == "not_successful_preparation"
    assert report["failure_observations"][0]["reason"] == "preparation_did_not_complete"
    assert "secret arbitrary message" not in json.dumps(report)


@pytest.mark.parametrize("updates", [{"observed_at": "2026-10-17T09:00:00+09:00"},
                                    {"observed_at": "2026-10-17"}, {"reused_preparation": 1},
                                    {"format": "other"}, {"spec_identity": "invalid"},
                                    {"extra": "unsupported"}])
def test_malformed_observation_schema_rejects_latency(run, updates):
    observation(run, **updates)
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] is None and item["missing_reason"] == "invalid_preparation_observation"


@pytest.mark.parametrize("raw", ['{"status":"a","status":"b"}', '{"local_call_seconds":NaN}', '{"format":'])
def test_corrupt_or_truncated_observation_is_a_separate_missing_sample(run, raw):
    first = observation(run)
    broken = observation(run, 2)
    broken.write_text(raw, encoding="utf-8")
    raw_samples = samples(owner.report_experiment_timings(run), "local_preparation")
    assert len(raw_samples) == 2 and raw_samples[0]["source"] == str(first)
    assert raw_samples[0]["seconds"] == 12.5
    assert raw_samples[1]["seconds"] is None
    assert raw_samples[1]["missing_reason"] == "invalid_json_or_object"


def test_oversized_json_is_not_read_or_treated_as_zero(run):
    path = observation(run)
    path.write_bytes(b" " * (owner.MAX_JSON_BYTES + 1))
    item = samples(owner.report_experiment_timings(run), "local_preparation")[0]
    assert item["seconds"] is None and item["missing_reason"] == "json_size_limit_exceeded"


def test_observation_enumeration_cap_is_explicit_and_limits_report(run, monkeypatch):
    monkeypatch.setattr(owner, "MAX_OBSERVATION_ENTRIES", 2)
    for index in range(1, 5):
        observation(run, index)
    report = owner.report_experiment_timings(run)
    assert len(samples(report, "local_preparation")) == 2
    assert any(d["reason"] == "observation_enumeration_truncated" and d["entry_limit"] == 2 for d in report["diagnostics"])


def test_reader_is_allowlisted_read_only_and_does_not_read_credentials(run, monkeypatch):
    observation(run)
    write(run / "local_observations/credentials.json", {"key": "never-read"})
    write(run / "operations/credentials.json", {"key": "never-read"})
    write(run / "prepared/package/source_files.json", {"path": "credentials.json"})
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    original = Path.open
    opened = []
    def guarded(path, mode="r", *args, **kwargs):
        assert "w" not in mode and "a" not in mode and "+" not in mode
        assert path.name not in {"credentials.json", "source_files.json"}
        opened.append(path)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    report = owner.report_experiment_timings(run)
    assert opened and "never-read" not in json.dumps(report)
    monkeypatch.setattr(Path, "open", original)
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_symlink_observation_and_ancestor_are_rejected(run, tmp_path):
    source = write(tmp_path / "outside.json", {"key": "never-read"})
    destination = run / "local_observations" / ("preparation-" + "1" * 32 + ".json")
    destination.parent.mkdir()
    try:
        destination.symlink_to(source)
    except OSError:
        pytest.skip("Host does not permit creating synthetic symlinks")
    with pytest.raises(ValueError, match="Symlink"):
        owner.report_experiment_timings(run)
    alias = tmp_path / "run-alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        owner.report_experiment_timings(alias)


def test_cli_emits_report_and_safe_error_without_writes(run, capsys):
    observation(run)
    assert owner.main(["--run-directory", str(run)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "OBSERVATIONS_ONLY"
    assert owner.main(["--run-directory", str(run / "..")]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error == {"status": "ERROR", "reason": "invalid_or_unsafe_run_directory"}
