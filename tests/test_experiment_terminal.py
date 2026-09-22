"""Synthetic descriptor binding tests; no models, protected roles or remote work."""
import copy
import hashlib
import json
from unittest.mock import Mock

import pytest

from molgap.experiment_spec import ExperimentSpec, TERMINAL_PROTOCOL
from molgap.experiment_terminal import (
    TerminalDescriptor, execute_terminal_descriptor, translate_terminal_descriptor,
)
from molgap.research_memory import terminal_wiring
from molgap.research_memory.trace import canonicalize_trace
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import payload
from test_terminal_trace_closure import create_candidate_arm


def fact(value=None, reason="Not retained by synthetic fixture"):
    return {"value": value, "missing_reason": reason if value is None else None}


def observations(spec, arm):
    declaration = spec.to_dict()
    return {
        "identity": {
            "experiment_id": declaration["experiment_id"], "logical_run_id": declaration["logical_run_id"],
            "arm_id": arm["arm_id"], "spec_identity": spec.identity,
            "attempt_id": fact("att-1"),
            "platform": {"name": declaration["platform"]["name"], "run_reference": fact()},
            "family": fact(copy.deepcopy(arm["family"])),
            "recipe_identity": fact(canonical_fingerprint(arm["training"]["recipe"])),
            "source_commit": fact("1" * 40), "source_package_sha256": fact(),
            "data_identity": fact(canonical_fingerprint(arm["data"])),
            "split_identity": fact(canonical_fingerprint({"split": arm["data"]["split"], "roles": arm["data"]["roles"]})),
            "feature_identity": fact(arm["data"]["feature_sha256"]), "target": fact(arm["data"]["target"]),
            "initialization_identity": fact(canonical_fingerprint(arm["initialization"])),
        },
        "terminal": {"status": fact(), "exit_reason": fact()},
        "artifacts": {name: {"status": "missing", "locator": None, "sha256": None,
                             "missing_reason": "Not bound in synthetic descriptor"}
                      for name in ("metrics", "predictions", "checkpoint", "trace")},
        "progress": {name: fact() for name in ("epoch", "step", "samples")},
        "costs": [{"metric": "device_time", "unit": None, "value": None,
                   "status": "measurement_missing", "reason": "No measurement retained"}],
        "missing_evidence": ["Synthetic fixture has no runtime observation receipt"],
    }


@pytest.fixture
def case(tmp_path, payload):
    spec = ExperimentSpec(payload)
    arms = []
    for index, arm in enumerate(payload["arms"]):
        trajectory_id, run_id = f"TB-descriptor-{index}", f"descriptor-run-{index}"
        folder = f"descriptor_{index}"
        create_candidate_arm(tmp_path, folder, trajectory_id, run_id, f"ev-descriptor-{index}")
        arms.append({
            "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
            "trajectory_id": trajectory_id, "run_id": run_id,
            "trajectory": f"experiments/{folder}/trajectory.json",
            "terminal": f"experiments/{folder}/terminal.json",
            "observed": observations(spec, arm),
        })
    return spec, {
        "schema_version": TERMINAL_PROTOCOL, "spec_identity": spec.identity,
        "experiment_id": payload["experiment_id"], "logical_run_id": payload["logical_run_id"],
        "arms": arms,
    }


def translate(tmp_path, case):
    spec, data = case
    return translate_terminal_descriptor(tmp_path, spec, TerminalDescriptor(spec, data))


def rewrite(path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_translation_is_exact_and_keeps_descriptor_order(tmp_path, case):
    case[1]["arms"].reverse()
    assert translate(tmp_path, case) == [
        {"arm_identifier": arm["arm_id"], "trajectory": arm["trajectory"], "terminal": arm["terminal"]}
        for arm in case[1]["arms"]
    ]


def test_canonical_roundtrip_and_detachment(case):
    spec, data = case
    descriptor = TerminalDescriptor(spec, data)
    canonical = descriptor.to_json()
    assert canonical == json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert TerminalDescriptor.from_json(spec, json.dumps(data, indent=4)) == descriptor
    assert TerminalDescriptor(spec, dict(reversed(list(data.items())))) == descriptor
    descriptor.to_dict()["arms"].clear()
    data["arms"].clear()
    assert descriptor.to_json() == canonical


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_json_fields_rejected(case, nested):
    spec, data = case
    field = "arm_id" if nested else "spec_identity"
    value = data["arms"][0][field] if nested else data[field]
    token = json.dumps(field) + ": " + json.dumps(value)
    text = json.dumps(data).replace(token, token + ", " + token, 1)
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        TerminalDescriptor.from_json(spec, text)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field,value", [
    ("unknown", None), ("ready_for_desktop", True), ("replay_ready", True),
    ("executor", "arbitrary.module:call"), ("callback", lambda: None),
])
def test_unknown_fields_and_execution_hooks_rejected(case, nested, field, value):
    spec, data = case
    (data["arms"][0] if nested else data)[field] = value
    with pytest.raises(ValueError, match="missing or unknown fields"):
        TerminalDescriptor(spec, data)


@pytest.mark.parametrize("field", ["schema_version", "spec_identity", "experiment_id", "logical_run_id"])
def test_wrong_spec_identity_and_protocol(case, field):
    spec, data = case
    data[field] = "0" * 64 if field == "spec_identity" else "wrong"
    with pytest.raises(ValueError):
        TerminalDescriptor(spec, data)


def test_descriptor_cannot_be_reused_with_changed_spec(tmp_path, case):
    spec, data = case
    descriptor = TerminalDescriptor(spec, data)
    changed = spec.to_dict()
    changed["prospective"]["stop_rule"] = "A different stop rule"
    with pytest.raises(ValueError, match="spec identity"):
        translate_terminal_descriptor(tmp_path, ExperimentSpec(changed), descriptor)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "wrong_digest"])
def test_arm_membership_and_identity(case, change):
    spec, data = case
    if change == "missing":
        data["arms"].pop()
    elif change == "duplicate":
        data["arms"][1] = copy.deepcopy(data["arms"][0])
    elif change == "unknown":
        data["arms"][0]["arm_id"] = "unknown"
    else:
        data["arms"][0]["arm_identity"] = "0" * 64
    with pytest.raises(ValueError):
        TerminalDescriptor(spec, data)


@pytest.mark.parametrize("field", ["trajectory_id", "trajectory", "terminal"])
def test_duplicate_mapping_rejected(case, field):
    spec, data = case
    data["arms"][1][field] = data["arms"][0][field]
    with pytest.raises(ValueError, match="duplicate"):
        TerminalDescriptor(spec, data)


@pytest.mark.parametrize("field", ["trajectory_id", "run_id", "action_id"])
def test_terminal_cross_links_rejected(tmp_path, case, field):
    path = tmp_path / case[1]["arms"][0]["terminal"]
    rewrite(path, lambda value: value.update({field: "wrong"}))
    with pytest.raises(ValueError, match="identity mismatch|not frozen"):
        translate(tmp_path, case)


def test_swapped_arm_files_rejected(tmp_path, case):
    arms = case[1]["arms"]
    arms[0]["terminal"], arms[1]["terminal"] = arms[1]["terminal"], arms[0]["terminal"]
    with pytest.raises(ValueError, match="identity mismatch"):
        translate(tmp_path, case)


def test_existing_trajectory_schema_is_enforced(tmp_path, case):
    rewrite(tmp_path / case[1]["arms"][0]["trajectory"], lambda value: value.pop("hypothesis"))
    with pytest.raises(ValueError):
        translate(tmp_path, case)


@pytest.mark.parametrize("field", ["trajectory", "terminal", "trace", "trace_source", "canonical_trace_output", "recovery_spec"])
@pytest.mark.parametrize("path", [
    "../outside.json", "experiments/../outside.json", "experiments\\..\\outside.json",
    "/tmp/outside.json", "C:/outside.json", "C:outside.json", "//host/file.json",
    "https://example.com/file.json", "experiments/file.json:stream", "./local.json",
    "experiments//local.json", "experiments/NUL", "experiments/file. ", None,
])
def test_unsafe_or_noncanonical_paths_rejected(case, field, path):
    spec, data = case
    data["arms"][0][field] = path
    with pytest.raises(ValueError):
        TerminalDescriptor(spec, data)


def test_symlink_escape_rejected(tmp_path, case):
    outside = tmp_path.parent / (tmp_path.name + "_outside.json")
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    case[1]["arms"][0]["trace_source"] = "linked.json"
    with pytest.raises(ValueError, match="escapes repository"):
        translate(tmp_path, case)


def test_finalization_directory_collision_rejected(tmp_path, case):
    arms = case[1]["arms"]
    path = arms[0]["trajectory"].replace("trajectory.json", "second_trajectory.json")
    (tmp_path / path).write_bytes((tmp_path / arms[1]["trajectory"]).read_bytes())
    arms[1]["trajectory"] = path
    with pytest.raises(ValueError, match="share an RML finalization directory"):
        translate(tmp_path, case)


@pytest.mark.parametrize("field", ["trajectory", "terminal", "trace", "trace_source", "recovery_spec"])
def test_explicit_missing_inputs_rejected(tmp_path, case, field):
    case[1]["arms"][0][field] = "experiments/missing.json"
    with pytest.raises(ValueError, match="missing descriptor input file"):
        translate(tmp_path, case)


def add_optional_files(tmp_path, case):
    arm = case[1]["arms"][0]
    folder = arm["terminal"].rsplit("/", 1)[0]
    arm.update(trace=folder + "/canonical_trace.json", trace_source=folder + "/raw_trace.json",
               canonical_trace_output=folder + "/recovered_trace.json", recovery_spec=folder + "/recovery.json")
    trace = canonicalize_trace({"trajectory_id": arm["trajectory_id"], "run_id": arm["run_id"]})
    recovery = {"metric_semantics": trace["metric_semantics"], "field_mapping": {"optimizer_step": "step"}}
    (tmp_path / arm["trace"]).write_text(json.dumps(trace), encoding="utf-8")
    (tmp_path / arm["recovery_spec"]).write_text(json.dumps(recovery), encoding="utf-8")
    return arm, recovery


def test_optional_paths_and_unknown_evidence_preserved(tmp_path, case):
    arm, recovery = add_optional_files(tmp_path, case)
    descriptor = TerminalDescriptor(case[0], case[1])
    result = translate_terminal_descriptor(tmp_path, case[0], descriptor)
    for field in ("trace", "trace_source", "canonical_trace_output"):
        assert result[0][field] == arm[field]
    assert result[0]["recovery_spec"] == recovery
    assert descriptor.to_dict()["arms"][0]["recovery_spec"] == arm["recovery_spec"]
    assert result[0]["recovery_spec"]["metric_semantics"]["ema_dev_metric"] is None
    assert "trajectory_id" not in result[0]["recovery_spec"]
    assert not (tmp_path / arm["canonical_trace_output"]).exists()


@pytest.mark.parametrize("field", ["trajectory_id", "run_id"])
def test_cross_arm_canonical_trace_rejected(tmp_path, case, field):
    arm, _ = add_optional_files(tmp_path, case)
    rewrite(tmp_path / arm["trace"], lambda value: value.update({field: "wrong"}))
    with pytest.raises(ValueError, match="trace identity mismatch"):
        translate(tmp_path, case)


@pytest.mark.parametrize("change", ["unknown", "identity", "mapping", "duplicate", "nan"])
def test_recovery_spec_is_strict(tmp_path, case, change):
    arm, recovery = add_optional_files(tmp_path, case)
    path = tmp_path / arm["recovery_spec"]
    if change == "unknown":
        recovery["parser"] = "arbitrary.module:call"
    elif change == "identity":
        recovery["run_id"] = "wrong"
    elif change == "mapping":
        recovery["field_mapping"]["unknown"] = "column"
    elif change == "nan":
        recovery["field_mapping"]["optimizer_step"] = float("nan")
    text = json.dumps(recovery)
    if change == "duplicate":
        text = text.replace('"field_mapping":', '"field_mapping": {}, "field_mapping":')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        translate(tmp_path, case)


def test_output_cannot_overwrite_input(tmp_path, case):
    arm = case[1]["arms"][0]
    arm["canonical_trace_output"] = arm["terminal"]
    with pytest.raises(ValueError, match="aliases an input"):
        translate(tmp_path, case)


def test_translation_never_executes_or_changes_files(tmp_path, case, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("validation executed a mutating workflow"))
    for name in ("close_terminal_multi_arm", "close_terminal_arm", "resolve_trace_for_terminal_arm",
                 "finalize_rebuild_backtest", "recover_trace", "build_default_trace_manifest"):
        monkeypatch.setattr(terminal_wiring, name, forbidden)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    translate(tmp_path, case)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    forbidden.assert_not_called()


def test_execution_is_only_existing_multi_arm_call(tmp_path, case, monkeypatch):
    expected = translate(tmp_path, case)
    results = [{"pipeline_status": "EXISTING_API_RESULT"}]
    close = Mock(return_value=results)
    monkeypatch.setattr(terminal_wiring, "close_terminal_multi_arm", close)
    assert execute_terminal_descriptor(tmp_path, case[0], TerminalDescriptor(*case)) is results
    close.assert_called_once_with(tmp_path, expected)


def test_existing_retained_trace_guard_still_fails_closed(tmp_path, case):
    # Omitting the descriptor trace does not waive retained evidence obligations.
    raw = tmp_path / "experiments/descriptor_0/raw_trace.json"
    raw.unlink()
    descriptor = TerminalDescriptor(*case)
    translate_terminal_descriptor(tmp_path, case[0], descriptor)
    with pytest.raises(FileNotFoundError, match="FAIL CLOSED: declared retained trace file missing"):
        execute_terminal_descriptor(tmp_path, case[0], descriptor)
    assert not list(tmp_path.rglob("rml_finalized"))


@pytest.mark.parametrize("path", [
    (), ("identity",), ("identity", "family"), ("identity", "family", "value"),
    ("identity", "platform"), ("identity", "platform", "run_reference"),
    ("terminal",), ("terminal", "status"), ("artifacts",), ("artifacts", "metrics"),
    ("progress",), ("progress", "step"), ("costs", 0),
])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_observed_nested_exact_fields(case, path, change):
    spec, data = case
    value = data["arms"][0]["observed"]
    for field in path:
        value = value[field]
    if change == "extra":
        value["ready"] = True
    else:
        value.pop(next(iter(value)))
    with pytest.raises(ValueError):
        TerminalDescriptor(spec, data)


@pytest.mark.parametrize("field", ["experiment_id", "logical_run_id", "arm_id", "spec_identity"])
def test_per_arm_binding_mismatch(case, field):
    case[1]["arms"][0]["observed"]["identity"][field] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("field", [
    "family", "recipe_identity", "data_identity", "split_identity", "feature_identity", "target", "initialization_identity",
])
@pytest.mark.parametrize("value", ["wrong", "A" * 64, True, {"name": "gptrans_t", "version": "2"}])
def test_observed_spec_identity_mismatch(case, field, value):
    case[1]["arms"][0]["observed"]["identity"][field] = fact(value)
    with pytest.raises(ValueError, match="identity mismatch"):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("field,value", [
    ("attempt_id", "../attempt"), ("source_commit", "ABCDEF"),
    ("source_commit", "A" * 40), ("source_commit", "1" * 39),
    ("source_package_sha256", "A" * 64), ("source_package_sha256", "1" * 63),
])
def test_observed_identity_syntax(case, field, value):
    case[1]["arms"][0]["observed"]["identity"][field] = fact(value)
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("field,value", [("attempt_id", "other"), ("source_commit", "2" * 40)])
def test_observed_action_identity_mismatch(tmp_path, case, field, value):
    case[1]["arms"][0]["observed"]["identity"][field] = fact(value)
    with pytest.raises(ValueError, match="identity mismatch with frozen action"):
        translate(tmp_path, case)


def test_observed_physical_platform_binding(case):
    case[1]["arms"][0]["observed"]["identity"]["platform"]["name"] = "ims"
    with pytest.raises(ValueError, match="identity mismatch"):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("value", [fact(None, ""), fact(None, None), {"value": 1, "missing_reason": "Unknown"}])
def test_unknown_requires_reason_and_known_forbids_reason(case, value):
    case[1]["arms"][0]["observed"]["progress"]["step"] = value
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


def test_unknown_identity_and_terminal_remain_unknown(tmp_path, case):
    observed = case[1]["arms"][0]["observed"]
    for name in ("attempt_id", "family", "recipe_identity", "source_commit", "source_package_sha256",
                 "data_identity", "split_identity", "feature_identity", "target", "initialization_identity"):
        observed["identity"][name] = fact()
    descriptor = TerminalDescriptor(*case)
    assert descriptor.to_dict()["arms"][0]["observed"] == observed
    result = translate_terminal_descriptor(tmp_path, case[0], descriptor)
    assert set(result[0]) == {"arm_identifier", "trajectory", "terminal"}
    assert descriptor.to_dict()["arms"][0]["observed"]["terminal"]["status"]["value"] is None


@pytest.mark.parametrize("status", ["complete", "failed", "cancelled", "interrupted"])
def test_explicit_terminal_observations_preserved(case, status):
    observed = case[1]["arms"][0]["observed"]
    observed["terminal"] = {"status": fact(status), "exit_reason": fact("Observed scheduler exit")}
    assert TerminalDescriptor(*case).to_dict()["arms"][0]["observed"]["terminal"] == observed["terminal"]


@pytest.mark.parametrize("status", ["ready", "running", "unknown", True, 0])
def test_invalid_terminal_status(case, status):
    case[1]["arms"][0]["observed"]["terminal"]["status"] = fact(status)
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


def available_artifact(tmp_path, case, name):
    locator = "experiments/descriptor_0/observed_" + name + ".bin"
    content = b"synthetic artifact bytes"
    (tmp_path / locator).write_bytes(content)
    artifact = {"status": "available", "locator": locator,
                "sha256": hashlib.sha256(content).hexdigest(), "missing_reason": None}
    case[1]["arms"][0]["observed"]["artifacts"][name] = artifact
    return artifact


@pytest.mark.parametrize("name", ["metrics", "predictions", "checkpoint", "trace"])
def test_available_artifact_bytes_bound_read_only(tmp_path, case, name):
    artifact = available_artifact(tmp_path, case, name)
    before = (tmp_path / artifact["locator"]).read_bytes()
    translate(tmp_path, case)
    assert (tmp_path / artifact["locator"]).read_bytes() == before
    (tmp_path / artifact["locator"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA mismatch"):
        translate(tmp_path, case)


@pytest.mark.parametrize("change", ["missing_file", "uppercase", "short_sha", "null_sha", "reason", "null_locator"])
def test_invalid_available_artifact(tmp_path, case, change):
    artifact = available_artifact(tmp_path, case, "metrics")
    if change == "missing_file":
        (tmp_path / artifact["locator"]).unlink()
    elif change == "uppercase":
        artifact["sha256"] = "A" * 64
    elif change == "short_sha":
        artifact["sha256"] = "a" * 63
    elif change == "null_sha":
        artifact["sha256"] = None
    elif change == "reason":
        artifact["missing_reason"] = "Contradiction"
    else:
        artifact["locator"] = None
    with pytest.raises(ValueError):
        translate(tmp_path, case)


def test_explicit_missing_artifacts_not_required(tmp_path, case):
    artifact = case[1]["arms"][0]["observed"]["artifacts"]["checkpoint"]
    artifact["locator"] = "experiments/not_retained.pt"
    translate(tmp_path, case)
    assert not (tmp_path / artifact["locator"]).exists()


@pytest.mark.parametrize("field,value", [("status", "ready"), ("sha256", "a" * 64), ("missing_reason", None)])
def test_invalid_missing_artifact(case, field, value):
    case[1]["arms"][0]["observed"]["artifacts"]["metrics"][field] = value
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("locator", ["../escape", "C:/escape", "https://host/file", "a\\b", "a/../b"])
def test_artifact_locator_boundary(case, locator):
    case[1]["arms"][0]["observed"]["artifacts"]["trace"]["locator"] = locator
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


def test_output_cannot_overwrite_observed_artifact(tmp_path, case):
    artifact = available_artifact(tmp_path, case, "metrics")
    case[1]["arms"][0]["canonical_trace_output"] = artifact["locator"]
    with pytest.raises(ValueError, match="aliases an input"):
        translate(tmp_path, case)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["epoch", "step", "samples"])
def test_progress_rejects_non_integer_observations(case, field, value):
    case[1]["arms"][0]["observed"]["progress"][field] = fact(value)
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("status", ["measured", "estimated", "measurement_missing"])
def test_native_cost_and_zero_progress_preserved(case, status):
    observed = case[1]["arms"][0]["observed"]
    observed["progress"]["step"] = fact(0)
    cost = {"metric": "device_time", "unit": "DCU-hours", "status": status,
            "value": None if status == "measurement_missing" else 0,
            "reason": None if status == "measured" else "Explicit measurement absence or estimate basis"}
    observed["costs"] = [cost]
    restored = TerminalDescriptor(*case).to_dict()["arms"][0]["observed"]
    assert restored["costs"] == [cost]
    assert restored["progress"]["step"] == fact(0)
    assert restored["progress"]["epoch"]["value"] is None


@pytest.mark.parametrize("field,value", [
    ("value", 0), ("value", False), ("reason", None), ("reason", ""), ("status", "unknown"),
])
def test_missing_cost_cannot_be_guessed(case, field, value):
    case[1]["arms"][0]["observed"]["costs"][0][field] = value
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("status", ["measured", "estimated"])
@pytest.mark.parametrize("value", [None, -1, True, "1", float("nan"), float("inf")])
def test_cost_numeric_validation(case, status, value):
    case[1]["arms"][0]["observed"]["costs"] = [{
        "metric": "device_time", "unit": "GPU-seconds", "status": status,
        "value": value, "reason": None if status == "measured" else "Observed rate estimate",
    }]
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("change", ["empty", "duplicate", "unknown_unit", "estimate_without_basis"])
def test_cost_evidence_required(case, change):
    costs = case[1]["arms"][0]["observed"]["costs"]
    if change == "empty":
        costs.clear()
    elif change == "duplicate":
        costs.append(copy.deepcopy(costs[0]))
    else:
        costs[0].update(status="estimated", value=1, unit="seconds", reason="Estimate basis")
        costs[0]["unit" if change == "unknown_unit" else "reason"] = None
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


@pytest.mark.parametrize("value", [None, "missing", [None], [""], [{}]])
def test_missing_evidence_reasons_are_explicit(case, value):
    case[1]["arms"][0]["observed"]["missing_evidence"] = value
    with pytest.raises(ValueError):
        TerminalDescriptor(*case)


def test_observed_metadata_cannot_be_swapped_between_arms(case):
    arms = case[1]["arms"]
    arms[0]["observed"], arms[1]["observed"] = arms[1]["observed"], arms[0]["observed"]
    with pytest.raises(ValueError, match="identity mismatch"):
        TerminalDescriptor(*case)


def test_observation_duplicate_json_key_rejected(case):
    encoded = json.dumps(case[1]).replace('"epoch":', '"epoch": null, "epoch":', 1)
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        TerminalDescriptor.from_json(case[0], encoded)


@pytest.mark.parametrize("status", ["available", "missing"])
def test_observed_artifact_symlink_escape_rejected(tmp_path, case, status):
    outside = tmp_path.parent / (tmp_path.name + "_artifact.bin")
    outside.write_bytes(b"outside")
    link = tmp_path / "linked_artifact.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this host")
    case[1]["arms"][0]["observed"]["artifacts"]["metrics"] = {
        "status": status, "locator": "linked_artifact.bin",
        "sha256": hashlib.sha256(b"outside").hexdigest() if status == "available" else None,
        "missing_reason": None if status == "available" else "Not retained",
    }
    with pytest.raises(ValueError, match="escapes repository"):
        translate(tmp_path, case)


def test_complete_does_not_supply_unknown_exit_or_progress(tmp_path, case):
    observed = case[1]["arms"][0]["observed"]
    observed["terminal"]["status"] = fact("complete")
    descriptor = TerminalDescriptor(*case)
    translate_terminal_descriptor(tmp_path, case[0], descriptor)
    restored = descriptor.to_dict()["arms"][0]["observed"]
    assert restored == observed
    assert restored["terminal"]["exit_reason"]["value"] is None
    assert restored["progress"]["step"]["value"] is None
    assert restored["costs"][0]["status"] == "measurement_missing"
