"""Synthetic descriptor binding tests; no models, protected roles or remote work."""
import copy
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
    with pytest.raises(ValueError, match="FAIL CLOSED"):
        execute_terminal_descriptor(tmp_path, case[0], descriptor)
    assert not list(tmp_path.rglob("rml_finalized"))
