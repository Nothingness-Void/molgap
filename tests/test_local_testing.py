"""Flow selection must neither duplicate pytest work nor hide its exit status."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from molgap import local_testing


def test_combined_flows_run_one_session_without_duplicate_files(monkeypatch):
    calls = []
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 1)
    assert local_testing.main([
        "--scope", "prepare", "--scope", "submit", "--scope", "prepare", "--", "-q",
    ]) == 1
    assert len(calls) == 1
    files = calls[0][:-1]
    assert calls[0][-1] == "-q"
    assert len(files) == len(set(files))
    assert files.count("tests/test_reform_workflow.py") == 1
    assert "tests/test_kaggle_submission.py" in files


def test_list_is_read_only_and_all_selections_resolve(monkeypatch, capsys):
    monkeypatch.setattr(pytest, "main", lambda args: pytest.fail("listing ran tests"))
    assert local_testing.main(["--list"]) == 0
    groups = json.loads(capsys.readouterr().out)
    root = Path(__file__).resolve().parents[1]
    assert all((root / "tests" / (name + ".py")).is_file()
               for names in groups.values() for name in names)
    assert local_testing.main(["--scope", "prepare", "--scope", "submit", "--list"]) == 0
    files = json.loads(capsys.readouterr().out)
    assert len(files) == len(set(files))


def test_no_implicit_full_suite():
    with pytest.raises(SystemExit) as error:
        local_testing.main([])
    assert error.value.code == 2


def test_scopes_cover_existing_lifecycle_regressions():
    assert "test_experiment_run" in local_testing.SCOPES["prepare"]
    assert "test_reform_review" in local_testing.SCOPES["prepare"]
    assert "test_reform_catalog" in local_testing.SCOPES["prepare"]
    assert "test_reform_distribution" in local_testing.SCOPES["package"]
    assert "test_kaggle_family_retrieval" in local_testing.SCOPES["accept"]
    assert "test_experiment_cli" in local_testing.SCOPES["accept"]
    assert "test_modern2d_training_prepare" in local_testing.SCOPES["prepare"]
    assert "test_modern2d_qualification_prepare" in local_testing.SCOPES["prepare"]
    assert "test_modern2d_training_records" in local_testing.SCOPES["accept"]
    assert "test_modern2d_qualification_records" in local_testing.SCOPES["accept"]
    for scope in ("submit", "accept", "resume"):
        assert "test_reform_daily_platform" in local_testing.SCOPES[scope]


def test_test_inventory_keeps_worker_boundary_and_rejects_payloads(tmp_path, monkeypatch):
    from molgap import experiment_source_inventory as inventory
    from molgap.experiment_package import _name
    tracked = {*inventory._TEST_SUPPORT_FILES, "tests/conftest.py", "tests/test_data_namespace.py",
        "src/molgap/__init__.py", "src/molgap/data/loader.py", "src/molgap/local_runtime.py",
        "src/molgap/kaggle_credentials.py", "src/molgap/archive/old.py",
        "tests/cache.npz", "tests/data/manifest.json", "credentials.json", "data/weights.pt"}
    monkeypatch.setattr("molgap.v4_bundle._tracked_paths", lambda root: tracked)
    names = set(inventory.testing_source_files(tmp_path))
    assert {"tests/conftest.py", "tests/test_data_namespace.py", "pyproject.toml",
            "src/molgap/local_runtime.py", "src/molgap/kaggle_credentials.py"} <= names
    assert not {"tests/cache.npz", "tests/data/manifest.json", "credentials.json",
                "data/weights.pt", "src/molgap/archive/old.py"} & names
    runtime = set(inventory.runtime_source_files(tmp_path))
    assert not {"src/molgap/local_runtime.py", "src/molgap/kaggle_credentials.py"} & runtime
    assert not any(path.startswith("tests/") for path in runtime)
    for unsafe in ("tests/secret.py", "tests/credentials/helper.py", "tests/auth_token.py",
                   "tests/checkpoint.pt", "tests/data/manifest.json", "src/molgap/other_credentials.py"):
        with pytest.raises(ValueError):
            _name(unsafe)
    tracked.add("tests/secret.py")
    with pytest.raises(ValueError, match="sensitive"):
        inventory.testing_source_files(tmp_path)


def test_runtime_inventory_excludes_local_orchestration_but_test_inventory_keeps_it():
    from molgap.experiment_source_inventory import (
        _HOST_ONLY, runtime_source_files, testing_source_files)

    root = Path(__file__).resolve().parents[1]
    host_tools = {"src/molgap/local_testing.py", "src/molgap/local_storage.py",
                  "src/molgap/modern2d_training_prepare.py",
                  "src/molgap/modern2d_qualification_prepare.py",
                  "src/molgap/modern2d_research_prepare.py"}
    if (root / ".git").exists():
        runtime = set(runtime_source_files(root))
        testing = set(testing_source_files(root))
        assert host_tools.isdisjoint(runtime)
        assert host_tools <= testing
    else:
        # Extracted test bundles intentionally have no Git metadata. Their
        # inventory was fixed before extraction; verify the selected host tools
        # are present without trying to rebuild a checkout-bound inventory.
        assert host_tools <= _HOST_ONLY
        assert all((root / name).is_file() for name in host_tools)


def test_package_cli_uses_existing_bundle_and_lists_without_writing(tmp_path, monkeypatch, capsys):
    from molgap import experiment_source_inventory as inventory
    calls = []
    monkeypatch.setattr(inventory, "testing_source_files", lambda root: ["tests/conftest.py"])
    monkeypatch.setattr(inventory, "build_testing_source_package",
        lambda root, target: calls.append((root, target)) or {"file_count": 1})
    target = tmp_path / "package"
    assert local_testing.main(["--package", str(target), "--repo-root", str(tmp_path), "--list"]) == 0
    assert json.loads(capsys.readouterr().out) == ["tests/conftest.py"]
    assert not calls and not target.exists()
    assert local_testing.main(["--package", str(target), "--repo-root", str(tmp_path)]) == 0
    assert calls == [(tmp_path, target)]
    assert json.loads(capsys.readouterr().out) == {"file_count": 1}
    with pytest.raises(SystemExit):
        local_testing.main(["--package", str(target), "--scope", "runtime"])


def test_extracted_test_package_collects_all_flows_and_executes_its_own_tests(tmp_path):
    from molgap.experiment_source_inventory import (
        testing_source_files, build_testing_source_package, _TEST_SUPPORT_FILES)
    root = Path(__file__).resolve().parents[1]
    # Commit an independent source fixture, including this checkout's edits.
    # Never change the user's index or point its shared editable install here.
    source = tmp_path / "fixture-source"
    # An extracted distribution has no Git metadata. Build the fixture from
    # its source/test files, then let the real packager inspect the fixture Git.
    names = set(_TEST_SUPPORT_FILES)
    for directory in ("src/molgap", "tests"):
        names.update(path.relative_to(root).as_posix()
                     for path in (root / directory).rglob("*.py")
                     if "archive" not in path.relative_to(root).parts)
    for name in sorted(names):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes().replace(b"\r\n", b"\n"))
    commands = (("init",), ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "Test distribution fixture"),
        ("config", "core.autocrlf", "false"), ("add", "."),
        ("-c", "commit.gpgsign=false", "commit", "-m", "Frozen test distribution fixture"))
    for command in commands:
        subprocess.run(["git", *command], cwd=source, check=True, capture_output=True)
    package = tmp_path / "package"
    report = build_testing_source_package(source, package)
    assert report["file_count"] == len(testing_source_files(source))
    assert not (package / "experiment_spec.json").exists()
    with pytest.raises(ValueError, match="absent or empty"):
        build_testing_source_package(source, package)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(package / "source.tar.gz") as archive:
        # The bundle is generated from an already confined tracked inventory.
        archive.extractall(extracted)
    # Hide the original fixture entirely: children can use only extracted files
    # and the existing dependency interpreter. No checkout or .git is required.
    source.rename(tmp_path / "unavailable-source")
    program = r'''
import pathlib, sys
root=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(root/'src'))
from molgap import local_testing
if sys.argv[2]=='collect':
    args=[item for scope in local_testing.SCOPES for item in ('--scope',scope)]
    args += ['--','--collect-only','-q']
else:
    args=['--scope','runtime','--scope','prepare','--scope','accept','--','-q','-k',
          '(local_testing or reform_runtime or local_storage or machine_paths or '
          'execution_projection_tracks or parser_errors_are_finite_json) and not extracted_test_package']
code=local_testing.main(args)
assert code==0,code
for name,module in list(sys.modules.items()):
    if name=='molgap' or name.startswith('molgap.') or name.startswith('test_'):
        location=getattr(module,'__file__',None)
        if location is not None: assert pathlib.Path(location).is_relative_to(root),(name,location)
'''
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": "-1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for operation in ("collect", "execute"):
        child = subprocess.run([sys.executable, "-I", "-c", program, str(extracted), operation],
            cwd=extracted, env=environment, capture_output=True, text=True, timeout=90)
        assert child.returncode == 0, child.stdout + child.stderr
        assert ("collected" if operation == "collect" else "passed") in child.stdout
