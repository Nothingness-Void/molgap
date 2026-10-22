"""Local CLI coverage for later execution; never submit or execute a model."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import experiment_cli as cli
from molgap.experiment_launch import canonical_json
from molgap.experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2
from test_experiment_spec import payload
from test_experiment_terminal import case
from test_experiment_package import package, repo


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs/operations/examples"
SHA = "0" * 64


@pytest.fixture
def spec_file(tmp_path, payload):
    path = tmp_path / "spec.json"
    path.write_bytes(ExperimentSpec(payload).to_json().encode())
    return path


def invoke(capsys, args, expected=0):
    assert cli.main([str(arg) for arg in args]) == expected
    output = capsys.readouterr().out
    value = json.loads(output)
    assert output == canonical_json(value) + "\n"
    return value


@pytest.mark.parametrize("command", [None, "validate-spec", "package", "preflight",
                                     "run-diagnostic", "launch-receipt", "terminal",
                                     "plan-prospective", "check-release", "prepare-release", "check-preparation",
                                     "check-acceptance", "inspect-output", "accept-terminal",
                                     "sandbox", "source-dependencies"])
def test_help(capsys, command):
    with pytest.raises(SystemExit) as error:
        cli.main(([command] if command else []) + ["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert output == canonical_json(json.loads(output)) + "\n"
    assert "usage:" in json.loads(output)["help"]


@pytest.mark.parametrize("filename,family,addon", [
    ("gptrans_t_v1.json", "gptrans_t", None),
    ("k1_v1.json", "neural_atom_k1", "k1_pair_value"),
])
def test_examples(capsys, filename, family, addon):
    path = EXAMPLES / filename
    raw = path.read_bytes().decode()
    spec = ExperimentSpec.from_json(raw)
    assert raw == spec.to_json()
    assert ExperimentSpec.from_json(spec.to_json()) == spec
    result = invoke(capsys, ["validate-spec", "--spec", path])
    assert result == {"spec_identity": spec.identity, "spec": spec.to_dict()}
    arm = result["spec"]["arms"][0]
    assert arm["family"] == {"name": family, "version": "1"}
    assert arm["addons"] == ([] if addon is None else [
        {"name": addon, "version": "1", "config": {}, "source_sha256": SHA},
    ])
    assert "replay_ready" not in raw and "ready_for_desktop" not in raw
    assert "no data access authorized" in raw
    assert arm["data"]["dataset"]["sha256"] == SHA
    assert path.parent == EXAMPLES


def test_edge_state_example(capsys):
    path = EXAMPLES / "edge_state_v1.json"
    raw = path.read_bytes().decode()
    spec = ExperimentSpec.from_json(raw)
    assert raw == spec.to_json()
    result = invoke(capsys, ["validate-spec", "--spec", path])
    assert result["spec_identity"] == spec.identity
    reference, candidate = result["spec"]["arms"]
    assert reference["family"] == candidate["family"] == {
        "name": "edge_state_gps", "version": "1",
    }
    assert reference["addons"] == []
    assert candidate["addons"][0]["config"] == {"num_layers": 6}
    assert all(arm["initialization"]["state_sha256"] == SHA
               for arm in (reference, candidate))


@pytest.mark.parametrize("bad", ["{", '{"x":1,"x":2}', "{}", '{"replay_ready":true}'])
def test_bad_json(capsys, spec_file, bad):
    spec_file.write_bytes(bad.encode())
    assert invoke(capsys, ["validate-spec", "--spec", spec_file], 2)["status"] == "ERROR"


@pytest.mark.parametrize("change", ["newline", "pretty", "duplicate", "unknown"])
def test_noncanonical_and_unknown(capsys, spec_file, change):
    raw = spec_file.read_text()
    value = json.loads(raw)
    if change == "newline":
        raw += "\n"
    elif change == "pretty":
        raw = json.dumps(value, indent=2)
    elif change == "duplicate":
        raw = '{"logical_run_id":"hidden",' + raw[1:]
    else:
        value["callback"] = "arbitrary.module:execute"
        raw = canonical_json(value)
    spec_file.write_bytes(raw.encode())
    invoke(capsys, ["validate-spec", "--spec", spec_file], 2)


def test_invalid_local_paths(capsys, tmp_path):
    for path in (tmp_path / ".." / "escape.json", tmp_path / "missing.json"):
        invoke(capsys, ["validate-spec", "--spec", path], 2)


def test_package_passthrough(capsys, monkeypatch, spec_file, tmp_path):
    build = Mock(return_value={"package_identity": SHA})
    monkeypatch.setattr(cli, "build_experiment_source_package", build)
    invoke(capsys, ["package", "--spec", spec_file, "--repo-root", tmp_path,
                    "--output", tmp_path / "pkg", "--allowlist", "src/a.py", "README.md",
                    "--allowlist", "src/b.py"])
    build.assert_called_once_with(ExperimentSpec.from_json(spec_file.read_text()), tmp_path,
                                  ["src/a.py", "README.md", "src/b.py"], tmp_path / "pkg")


def test_release_command_reports_nonpass_and_rejects_duplicate_binding(capsys, monkeypatch, spec_file, tmp_path):
    monkeypatch.setattr(cli, "check_release_inputs", Mock(return_value={
        "status": "RELEASE_INPUTS_FAILED", "errors": [{"check": "initialization"}]}))
    args = ["check-release", "--spec", spec_file, "--package", tmp_path,
            "--expected-package-identity", SHA, "--recipe-file", "arm=recipe.json",
            "--required-module", "molgap.loader"]
    assert invoke(capsys, args, 1)["status"] == "RELEASE_INPUTS_FAILED"
    assert invoke(capsys, args + ["--recipe-file", "arm=other.json"], 2)["status"] == "ERROR"


@pytest.fixture
def v2_spec_file(tmp_path, payload):
    payload["schema_version"] = SCHEMA_VERSION_V2
    payload["prospective"] = {"arms": [
        {"arm_id": arm["arm_id"], "trajectory_id": f"trajectory-{index}",
         "plan_spec_ref": f"experiments/plan-inputs/arm-{index}.json",
         "plan_spec_sha256": SHA, "output": f"experiments/synthetic/arm-{index}"}
        for index, arm in enumerate(payload["arms"])
    ]}
    spec = ExperimentSpec(payload)
    path = tmp_path / "spec-v2.json"
    path.write_bytes(spec.to_json().encode())
    return path, spec


@pytest.mark.parametrize("status,code", [
    ("PROSPECTIVE_PLANNED_AND_RML_REBUILT", 0),
    ("PARTIAL_PROSPECTIVE_PLAN_REQUIRES_RECONCILIATION", 1),
])
def test_plan_prospective_passthrough(capsys, monkeypatch, tmp_path, v2_spec_file,
                                      status, code):
    path, spec = v2_spec_file
    planning = Mock(return_value=({"status": status}, code))
    monkeypatch.setattr(cli, "plan_prospective", planning)

    result = invoke(capsys, ["plan-prospective", "--spec", path,
                             "--repo-root", tmp_path], code)

    assert result["status"] == status
    planning.assert_called_once_with(spec, tmp_path)


def test_plan_prospective_keeps_library_output_off_stdout(
        capsys, monkeypatch, tmp_path, v2_spec_file):
    path, _ = v2_spec_file

    def noisy_plan(*_):
        print("planning diagnostic")
        return {"status": "PROSPECTIVE_PLANNED_AND_RML_REBUILT"}, 0

    monkeypatch.setattr(cli, "plan_prospective", noisy_plan)
    assert cli.main(["plan-prospective", "--spec", str(path),
                     "--repo-root", str(tmp_path)]) == 0
    output = capsys.readouterr()
    assert output.out == canonical_json({
        "status": "PROSPECTIVE_PLANNED_AND_RML_REBUILT",
    }) + "\n"
    assert output.err == "planning diagnostic\n"


@pytest.mark.parametrize("mutation", ["newline", "duplicate", "unknown"])
def test_plan_prospective_rejects_noncanonical_spec_before_dispatch(
        capsys, monkeypatch, tmp_path, v2_spec_file, mutation):
    path, _ = v2_spec_file
    raw = path.read_text()
    if mutation == "newline":
        raw += "\n"
    elif mutation == "duplicate":
        raw = '{"arms":[],"arms":[],' + raw[1:]
    else:
        value = json.loads(raw)
        value["submitter"] = "remote"
        raw = canonical_json(value)
    path.write_bytes(raw.encode())
    planning = Mock()
    monkeypatch.setattr(cli, "plan_prospective", planning)

    assert invoke(capsys, ["plan-prospective", "--spec", path,
                           "--repo-root", tmp_path], 2)["status"] == "ERROR"
    planning.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--submit"], ["--remote"], ["--callback", "module:run"],
    ["--repo-ro", "unused"],
])
def test_plan_prospective_rejects_execution_switches(
        capsys, monkeypatch, tmp_path, v2_spec_file, extra):
    path, _ = v2_spec_file
    planning = Mock()
    monkeypatch.setattr(cli, "plan_prospective", planning)

    assert invoke(capsys, ["plan-prospective", "--spec", path,
                           "--repo-root", tmp_path, *extra], 2)["status"] == "ERROR"
    planning.assert_not_called()


def test_v1_commands_remain_compatible_and_cannot_plan(
        capsys, monkeypatch, spec_file, tmp_path):
    from molgap import experiment_prospective as prospective

    spec = ExperimentSpec.from_json(spec_file.read_text())
    assert invoke(capsys, ["validate-spec", "--spec", spec_file]) == {
        "spec_identity": spec.identity, "spec": spec.to_dict(),
    }
    build = Mock(return_value={"package_identity": SHA})
    monkeypatch.setattr(cli, "build_experiment_source_package", build)
    assert invoke(capsys, ["package", "--spec", spec_file, "--repo-root", tmp_path,
                           "--output", tmp_path / "pkg", "--allowlist", "src/a.py"]) == {
        "package_identity": SHA,
    }
    build.assert_called_once_with(spec, tmp_path, ["src/a.py"], tmp_path / "pkg")

    planner = Mock()
    monkeypatch.setattr(prospective, "plan_many", planner)
    result = invoke(capsys, ["plan-prospective", "--spec", spec_file,
                             "--repo-root", tmp_path], 2)
    assert result["error"] == {
        "type": "ValueError", "message": "plan-prospective requires molgap-experiment-spec-v2",
    }
    planner.assert_not_called()


def test_package_rejects_escape_in_core(capsys, spec_file, repo, tmp_path):
    invoke(capsys, ["package", "--spec", spec_file, "--repo-root", repo,
                    "--output", tmp_path / "pkg", "--allowlist", "../escape.py"], 2)
    assert not (tmp_path / "pkg").exists()


@pytest.mark.parametrize("mode", ["loader-only-v1", "gptrans-model-smoke-v1"])
def test_preflight_passthrough(capsys, monkeypatch, spec_file, tmp_path, mode):
    shard = tmp_path / "shard"
    shard.mkdir()
    manifest = shard / "shard_manifest.json"
    manifest.write_bytes(b"{}")
    preflight = Mock(return_value={"status": "MISSING_REAL_SHARD"})
    monkeypatch.setattr(cli, "run_experiment_preflight", preflight)
    invoke(capsys, ["preflight", "--spec", spec_file, "--package", tmp_path / "pkg",
                    "--output", tmp_path / "out", "--shard-root", shard,
                    "--expected-package-identity", SHA,
                    "--expected-shard-manifest-sha256", "1" * 64, "--mode", mode], 1)
    preflight.assert_called_once_with(
        ExperimentSpec.from_json(spec_file.read_text()), tmp_path / "pkg", tmp_path / "out",
        expected_package_identity=SHA, shard_manifest=manifest, shard_root=shard,
        expected_shard_manifest_sha256="1" * 64, mode=mode,
    )


def test_real_core_missing_shard_no_worker(capsys, monkeypatch, package, spec_file, tmp_path):
    from molgap import experiment_preflight as preflight
    worker = Mock(side_effect=AssertionError("Missing real data must not run a worker"))
    monkeypatch.setattr(preflight, "_launch", worker)
    manifest = json.loads((package / "package_manifest.json").read_bytes())
    result = invoke(capsys, ["preflight", "--spec", spec_file, "--package", package,
                            "--output", tmp_path / "out", "--shard-root", tmp_path / "absent",
                            "--expected-package-identity", manifest["package_identity"],
                            "--expected-shard-manifest-sha256", SHA], 1)
    assert result["arms"][0]["status"] == "MISSING_REAL_SHARD"
    assert all(not arm["loader_batch_built"] for arm in result["arms"])
    worker.assert_not_called()


@pytest.mark.parametrize("worker", ["adapter_probe", "construct"])
@pytest.mark.parametrize("status,code", [("SUCCEEDED", 0), ("PARTIAL_FAILURE", 1),
                                        ("FAILED", 1), ("TIMED_OUT", 1)])
def test_runner_passthrough(capsys, monkeypatch, spec_file, tmp_path, worker, status, code):
    run = Mock(return_value={"status": status})
    monkeypatch.setattr(cli, "run_experiment", run)
    invoke(capsys, ["run-diagnostic", "--spec", spec_file, "--output", tmp_path / "out",
                    "--device", "0", "1", "--worker", worker], code)
    run.assert_called_once_with(ExperimentSpec.from_json(spec_file.read_text()),
                                tmp_path / "out", ["0", "1"], worker=worker)


@pytest.mark.parametrize("response", [False, True])
def test_receipt_passthrough(capsys, monkeypatch, spec_file, tmp_path, response):
    result = {"submission_state": "NOT_SUBMITTED", "submitter_status": "SUBMIT_UNIMPLEMENTED"}
    build, reconcile, write = Mock(return_value=result), Mock(return_value=result), Mock()
    monkeypatch.setattr(cli, "build_launch_receipt", build)
    monkeypatch.setattr(cli, "reconcile_platform_response", reconcile)
    monkeypatch.setattr(cli, "write_launch_receipt", write)
    args = ["launch-receipt", "--spec", spec_file, "--package", tmp_path / "pkg",
            "--expected-package-identity", SHA, "--output-dir", tmp_path]
    spec = ExperimentSpec.from_json(spec_file.read_text())
    if response:
        path = tmp_path / "response.json"
        path.write_bytes(b"{}")
        args += ["--response", path]
    assert invoke(capsys, args) == result
    if response:
        reconcile.assert_called_once_with(spec, tmp_path / "pkg", "{}", expected_package_identity=SHA)
        build.assert_not_called()
    else:
        build.assert_called_once_with(spec, tmp_path / "pkg", expected_package_identity=SHA)
        reconcile.assert_not_called()
    write.assert_called_once_with(canonical_json(result), tmp_path, spec, tmp_path / "pkg",
                                  expected_package_identity=SHA)


@pytest.mark.parametrize("bad", ['{"x":1,"x":2}', '{"unknown":true}', "{}\n"])
def test_response_validation_owned_by_core(capsys, package, spec_file, tmp_path, bad):
    pin = json.loads((package / "package_manifest.json").read_bytes())["package_identity"]
    response = tmp_path / "response.json"
    response.write_bytes(bad.encode())
    invoke(capsys, ["launch-receipt", "--spec", spec_file, "--package", package,
                    "--expected-package-identity", pin, "--output-dir", tmp_path,
                    "--response", response], 2)


@pytest.mark.parametrize("execute", [False, True])
def test_terminal_gate(capsys, monkeypatch, case, spec_file, tmp_path, execute):
    spec, value = case
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(canonical_json(value).encode())
    translate = Mock(return_value=[{"arm_identifier": "example"}])
    close = Mock(return_value=[{"pipeline_status": "COMPLETE"}])
    monkeypatch.setattr(cli, "translate_terminal_descriptor", translate)
    monkeypatch.setattr(cli, "execute_terminal_descriptor", close)
    args = ["terminal", "--spec", spec_file, "--descriptor", descriptor, "--repo-root", tmp_path]
    result = invoke(capsys, args + (["--execute"] if execute else []))
    assert result["executed"] is execute
    called, forbidden = (close, translate) if execute else (translate, close)
    called.assert_called_once_with(tmp_path, spec, cli.TerminalDescriptor(spec, value))
    forbidden.assert_not_called()


@pytest.mark.parametrize("mutation", ["path", "unknown", "duplicate", "newline"])
def test_terminal_fail_closed(capsys, monkeypatch, case, spec_file, tmp_path, mutation):
    spec, value = case
    close = Mock()
    monkeypatch.setattr(cli, "execute_terminal_descriptor", close)
    if mutation == "path":
        value["arms"][0]["trajectory"] = "../escape.json"
    elif mutation == "unknown":
        value["executor"] = "remote.submit"
    raw = canonical_json(value)
    if mutation == "duplicate":
        raw = '{"arms":[],' + raw[1:]
    elif mutation == "newline":
        raw += "\n"
    path = tmp_path / "descriptor.json"
    path.write_bytes(raw.encode())
    invoke(capsys, ["terminal", "--spec", spec_file, "--descriptor", path,
                    "--repo-root", tmp_path, "--execute"], 2)
    close.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--worker", "train"], ["--worker", "arbitrary.module:call"], ["--model", "x"],
    ["--train"], ["--remote"], ["--submit"], ["--callback", "x"],
    ["--replay-ready"], ["--ready-for-desktop"], ["--rml"], ["--work", "construct"],
])
def test_no_hidden_runner_switches(capsys, monkeypatch, spec_file, tmp_path, extra):
    run = Mock()
    monkeypatch.setattr(cli, "run_experiment", run)
    invoke(capsys, ["run-diagnostic", "--spec", spec_file, "--output", tmp_path / "out",
                    "--device", "cpu", *extra], 2)
    run.assert_not_called()


def test_execute_cannot_be_abbreviated(capsys, monkeypatch, spec_file, tmp_path):
    close = Mock()
    monkeypatch.setattr(cli, "execute_terminal_descriptor", close)
    invoke(capsys, ["terminal", "--spec", spec_file, "--descriptor", tmp_path / "absent",
                    "--repo-root", tmp_path, "--exec"], 2)
    close.assert_not_called()


def test_real_package_and_receipt_outputs(capsys, spec_file, repo, tmp_path):
    directory = tmp_path / "source-package"
    manifest = invoke(capsys, ["package", "--spec", spec_file, "--repo-root", repo,
                               "--output", directory, "--allowlist", "src/example.py"])
    assert manifest["relative_allowlist"] == ["src/example.py"]
    receipts = tmp_path / "receipt-snapshots"
    receipts.mkdir()
    result = invoke(capsys, ["launch-receipt", "--spec", spec_file, "--package", directory,
                            "--expected-package-identity", manifest["package_identity"],
                            "--output-dir", receipts])
    assert result["submission_state"] == "NOT_SUBMITTED"
    assert result["submitter_status"] == "SUBMIT_UNIMPLEMENTED"
    assert (receipts / (result["launch_identity"] + ".json")).read_bytes() == canonical_json(result).encode()


def test_real_terminal_translation_does_not_write(capsys, monkeypatch, case, spec_file, tmp_path):
    _, value = case
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(canonical_json(value).encode())
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    close = Mock(side_effect=AssertionError("Read-only terminal must not execute"))
    monkeypatch.setattr(cli, "execute_terminal_descriptor", close)
    result = invoke(capsys, ["terminal", "--spec", spec_file, "--descriptor", descriptor,
                            "--repo-root", tmp_path])
    assert result["executed"] is False
    assert len(result["arms"]) == len(value["arms"])
    assert before == {path.relative_to(tmp_path): path.read_bytes()
                      for path in tmp_path.rglob("*") if path.is_file()}
    close.assert_not_called()


def test_core_errors_are_not_swallowed(capsys, monkeypatch, spec_file, tmp_path):
    run = Mock(side_effect=RuntimeError("explicit core failure"))
    monkeypatch.setattr(cli, "run_experiment", run)
    result = invoke(capsys, ["run-diagnostic", "--spec", spec_file, "--output", tmp_path / "out",
                            "--device", "cpu"], 2)
    assert result["error"] == {"type": "RuntimeError", "message": "explicit core failure"}


def test_prepare_release_delegates_local_workflow(spec_file, tmp_path, capsys, monkeypatch):
    workflow_file = tmp_path / "workflow.json"
    workflow_file.write_text("{}")
    delegated = Mock(return_value=({"status": "LOCAL_PREPARATION_COMPLETE", "submitted": False}, 0))
    monkeypatch.setattr(cli, "prepare_experiment_release", delegated)
    result = invoke(capsys, ["prepare-release", "--spec", spec_file, "--workflow", workflow_file,
                             "--repo-root", tmp_path, "--output", tmp_path / "fresh"])
    assert result["submitted"] is False
    assert delegated.call_args.args[1:] == (tmp_path, "{}", tmp_path / "fresh")


def test_stage_only_acceptance_summary_keeps_retention_without_diagnosis(
        capsys, monkeypatch, tmp_path):
    from molgap import experiment_run

    accepted = Mock(return_value={
        "status": "MECHANICALLY_VERIFIED",
        "run_id": "synthetic-run",
        "attempt_id": "v1",
        "diagnosis": None,
        "terminal_staging": {"status": "STAGED"},
        "execution_retention": {"policy": "retained"},
        "arms": {},
    })
    monkeypatch.setattr(experiment_run, "accept_run_workflow", accepted)
    result = invoke(capsys, ["accept-workflow", "--run-id", "synthetic-run",
                             "--attempt-id", "v1", "--profile", tmp_path / "machine.json",
                             "--repo-root", tmp_path, "--stage-terminal"])
    assert result["status"] == "MECHANICALLY_VERIFIED"
    assert result["execution_retention"] == {"policy": "retained"}
    accepted.assert_called_once()


def test_build_preset_cli_passes_reference_components(capsys, monkeypatch, tmp_path):
    from molgap import experiment_presets

    repo = tmp_path / "repo"
    repo.mkdir()
    recipe = repo / "recipe.json"
    data_spec = repo / "data-spec.json"
    target = repo / "target.json"
    initial = repo / "initial.pt"
    reference = repo / "reference-plan.json"
    output = tmp_path / "preset.json"
    built = {"format": "molgap-scientific-baseline-preset-v1"}
    build = Mock(return_value=built)
    export = Mock()
    monkeypatch.setattr(experiment_presets, "build_baseline_preset", build)
    monkeypatch.setattr(experiment_presets, "export_baseline_preset", export)

    result = invoke(capsys, ["build-preset", "--repo-root", repo, "--name", "baseline",
                             "--family", "neural_atom_k1", "--family-version", "3",
                             "--version", "v2", "--recipe", recipe, "--data-spec", data_spec,
                             "--data-arm", "reference", "--target-manifest", target,
                             "--initial-state", initial, "--reference-plan", reference,
                             "--reference-arm", "reference", "--output", output])
    assert result == {"status": "PRESET_WRITTEN", "output": str(output)}
    build.assert_called_once_with(
        repo, name="baseline", version="v2", family=("neural_atom_k1", "3"),
        recipe=recipe, data_spec=data_spec, data_arm_id="reference",
        target_manifest=target, initial_state=initial, reference=reference,
        reference_arm_id="reference")
    export.assert_called_once_with(output, built)


def test_derive_input_cli_passes_repo_and_fresh_run_facts(capsys, monkeypatch, tmp_path):
    from molgap import experiment_presets

    repo = tmp_path / "repo"
    repo.mkdir()
    frozen_run = tmp_path / "frozen-run"
    profile = tmp_path / "machine.json"
    output = tmp_path / "input.json"
    template = {"format": "molgap-scientific-experiment-v1"}
    build = Mock(return_value=template)
    export = Mock()
    monkeypatch.setattr(experiment_presets, "build_scientific_input_template", build)
    monkeypatch.setattr(experiment_presets, "export_scientific_input_template", export)

    result = invoke(capsys, ["derive-input", "--from-run", frozen_run, "--repo-root", repo,
                             "--experiment-id", "question", "--run-id", "run-2",
                             "--attempt-id", "v1", "--profile", profile, "--output", output])
    assert result == {"status": "EDITABLE_DRAFT_WRITTEN", "output": str(output),
                      "authority": "REQUIRED_BEFORE_PREPARATION"}
    build.assert_called_once_with(
        frozen_run, repo, experiment_id="question", logical_run_id="run-2",
        attempt_id="v1", machine_profile=profile)
    export.assert_called_once_with(output, template, frozen_run=frozen_run)


def test_accept_workflow_preserves_role_audit_input_boundaries(
        capsys, monkeypatch, tmp_path):
    from molgap import experiment_run

    accepted = Mock(return_value={"status": "COMPLETE"})
    monkeypatch.setattr(experiment_run, "accept_run_workflow", accepted)
    base = ["accept-workflow", "--run-id", "run-1", "--attempt-id", "v1",
            "--profile", tmp_path / "machine.json", "--repo-root", tmp_path]

    single = tmp_path / "single.json"
    single.write_bytes(canonical_json({"context": {"arm_id": "baseline"}}).encode())
    invoke(capsys, [*base, "--role-audit", single])
    assert accepted.call_args.kwargs["role_audit"] == single

    mapped = tmp_path / "mapped.json"
    mapped.write_bytes(canonical_json({"baseline": {"context": {}}}).encode())
    accepted.reset_mock()
    invoke(capsys, [*base, "--role-audit", mapped])
    assert accepted.call_args.kwargs["role_audit"] == {"baseline": {"context": {}}}

    accepted.reset_mock()
    inline = '{"baseline":{"context":{}}}'
    invoke(capsys, [*base, "--role-audit", inline])
    assert accepted.call_args.kwargs["role_audit"] == {"baseline": {"context": {}}}


def test_accept_workflow_keeps_terminal_package_as_path(capsys, monkeypatch, tmp_path):
    from molgap import experiment_run

    accepted = Mock(return_value={"status": "COMPLETE"})
    monkeypatch.setattr(experiment_run, "accept_run_workflow", accepted)
    terminal = tmp_path / "terminal.json"
    terminal.write_bytes(canonical_json({
        "format": "molgap-rml-terminal-package-v1",
    }).encode())
    invoke(capsys, ["accept-workflow", "--run-id", "run-1", "--attempt-id", "v1",
                    "--profile", tmp_path / "machine.json", "--repo-root", tmp_path,
                    "--scientific-acceptance", terminal])
    assert accepted.call_args.kwargs["scientific_acceptance"] == terminal
