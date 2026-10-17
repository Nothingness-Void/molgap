"""Thin indexed-export CLI checks over generated CPU-synthetic fixtures only."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from test_pcqm_indexed_export import _files, _inputs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/modern2d_model_integration/scripts/export_inputs.py"
SPEC = importlib.util.spec_from_file_location("modern2d_export_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _request(args, boundary):
    return {
        "format": "molgap-indexed-private-export-request-v1",
        "scope": "CPU_SYNTHETIC_ONLY",
        "parent_root": args["parent_root"].relative_to(boundary).as_posix(),
        "output_root": args["output_root"].relative_to(boundary).as_posix(),
        "roles": list(args["roles"]),
        "permits": {role: asdict(permit) for role, permit in args["permits"].items()},
        "authorities": deepcopy(args["authorities"]),
        "role_histories": deepcopy(args["role_histories"]),
        "parent_acceptance_sha256": args["parent_acceptance_sha256"],
        "max_chunk_bytes": args["max_chunk_bytes"],
        "max_rows": dict(args["max_rows"]),
    }


def _write_request(boundary, request):
    path = boundary / "request.json"
    path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    return path


def _argv(request_path, boundary, *, execute=False):
    argv = ["--request", str(request_path), "--boundary", str(boundary)]
    if execute:
        argv.append("--execute")
    return argv


def test_export_inputs_help_is_a_thin_explicit_entrypoint(capsys):
    with pytest.raises(SystemExit) as error:
        CLI.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--request" in output and "--boundary" in output and "--execute" in output


def test_default_prepare_only_does_not_touch_synthetic_parent_or_output(tmp_path, capsys):
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    args = _inputs(boundary)
    request_path = _write_request(boundary, _request(args, boundary))
    before = _files(args["parent_root"])

    assert CLI.main(_argv(request_path, boundary)) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["status"] == "PREPARE_ONLY_NOT_EXECUTED"
    assert observed["remote_submission"] is False
    assert observed["training_released"] is False
    assert observed["roles"] == args["roles"]
    assert before == _files(args["parent_root"])
    assert not args["output_root"].exists()


@pytest.mark.parametrize("kind", ["traversal", "unknown_fields", "protected_permit"])
def test_export_inputs_rejects_confined_request_errors_before_export(tmp_path, kind):
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    args = _inputs(boundary)
    request = _request(args, boundary)
    execute = True
    if kind == "traversal":
        request["output_root"] = "../escape"
        match = "confined relative paths"
    elif kind == "unknown_fields":
        request["unexpected"] = True
        match = "Unknown selected-role export request"
    else:
        request["authorities"]["train_100k"]["protected"] = True
        match = "authority/history mismatch"
    request_path = _write_request(boundary, request)
    before = _files(args["parent_root"])

    with pytest.raises(ValueError, match=match):
        CLI.main(_argv(request_path, boundary, execute=execute))
    assert before == _files(args["parent_root"])
    assert not args["output_root"].exists()
    assert not (boundary.parent / "escape").exists()


def test_explicit_execute_exports_only_the_small_synthetic_fixture(tmp_path, capsys):
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    args = _inputs(boundary)
    request = _request(args, boundary)
    assert request["parent_root"] == "unified"
    assert request["output_root"] == "export"
    request_path = _write_request(boundary, request)

    assert CLI.main(_argv(request_path, boundary, execute=True)) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["status"] == "SYNTHETIC_DATA_ROLES_VERIFIED"
    assert observed["publication_status"] == "NOT_PUBLISHED"
    assert observed["training_released"] is False
    assert set(observed["roles"]) == set(args["roles"])
    assert (args["output_root"] / "acceptance.json").is_file()

    acceptance = json.loads((args["output_root"] / "acceptance.json").read_bytes())
    assert acceptance["scope"] == "CPU_SYNTHETIC_ONLY"
    assert acceptance["status"] == "SYNTHETIC_DATA_ROLES_VERIFIED"
    assert acceptance["training_authority"] == "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING"
    assert acceptance["publication_status"] == "NOT_PUBLISHED"
