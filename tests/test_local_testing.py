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
        "--scope", "train", "--scope", "resume", "--scope", "train", "--", "-q",
    ]) == 1
    assert len(calls) == 1
    files = calls[0][:-1]
    assert calls[0][-1] == "-q"
    assert len(files) == len(set(files))
    assert files.count(local_testing.SCOPES["train"][0]) == 1
    assert set(local_testing.SCOPES["resume"]) <= set(files)


def test_list_is_read_only_and_all_selections_resolve(monkeypatch, capsys):
    monkeypatch.setattr(pytest, "main", lambda args: pytest.fail("listing ran tests"))
    assert local_testing.main(["--list"]) == 0
    groups = json.loads(capsys.readouterr().out)
    root = Path(__file__).resolve().parents[1]
    assert all("::" in node and (root / node.split("::", 1)[0]).is_file()
               for nodes in groups.values() for node in nodes)
    assert local_testing.main(["--scope", "prepare", "--scope", "submit", "--list"]) == 0
    files = json.loads(capsys.readouterr().out)
    assert len(files) == len(set(files))


def test_no_implicit_full_suite():
    with pytest.raises(SystemExit) as error:
        local_testing.main([])
    assert error.value.code == 2


def test_test_inventory_keeps_worker_boundary_and_rejects_payloads(tmp_path, monkeypatch):
    from molgap import experiment_source_inventory as inventory
    from molgap.experiment_package import _name
    tracked = {*inventory._TEST_SUPPORT_FILES, "tests/conftest.py", "tests/test_data_namespace.py",
        "src/molgap/__init__.py", "src/molgap/data/loader.py", "src/molgap/local_runtime.py", "src/molgap/local_training.py",
        "src/molgap/kaggle_credentials.py", "src/molgap/archive/old.py",
        "research_memory/schemas/decision-state-v2.schema.json",
        "research_memory/policies/fixture-policy.1.json",
        "tests/cache.npz", "tests/data/manifest.json", "credentials.json", "data/weights.pt"}
    monkeypatch.setattr("molgap.v4_bundle._tracked_paths", lambda root: tracked)
    names = set(inventory.testing_source_files(tmp_path))
    assert {"tests/conftest.py", "tests/test_data_namespace.py", "pyproject.toml",
            "src/molgap/local_runtime.py", "src/molgap/local_training.py", "src/molgap/kaggle_credentials.py",
            "research_memory/schemas/decision-state-v2.schema.json",
            "research_memory/policies/fixture-policy.1.json"} <= names
    assert not {"tests/cache.npz", "tests/data/manifest.json", "credentials.json",
                "data/weights.pt", "src/molgap/archive/old.py"} & names
    runtime = set(inventory.runtime_source_files(tmp_path))
    assert not {"src/molgap/local_runtime.py", "src/molgap/local_training.py", "src/molgap/kaggle_credentials.py"} & runtime
    assert not any(path.startswith("tests/") for path in runtime)
    for unsafe in ("tests/secret.py", "tests/credentials/helper.py", "tests/auth_token.py",
                   "tests/checkpoint.pt", "tests/data/manifest.json", "src/molgap/other_credentials.py"):
        with pytest.raises(ValueError):
            _name(unsafe)
    tracked.add("tests/secret.py")
    with pytest.raises(ValueError, match="sensitive"):
        inventory.testing_source_files(tmp_path)


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
    for directory, suffix in (("src/molgap", ".py"), ("tests", ".py"),
                              ("research_memory/schemas", ".json"),
                              ("research_memory/policies", ".json")):
        names.update(path.relative_to(root).as_posix()
                     for path in (root / directory).rglob("*" + suffix)
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
    metadata = {name for name in names if name.endswith(".json") and
                name.startswith(("research_memory/schemas/", "research_memory/policies/"))}
    assert metadata <= set(testing_source_files(source))
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
    packaged_paths = {path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()}
    assert metadata <= packaged_paths
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
    # Exercise the extracted package once, without recursively running its
    # package-distribution test or selecting unrelated whole files.
    args=['--scope','runtime','--','-q']
    args += ['tests/test_ims_v4_submit.py::test_windows_packager_fills_source_and_contract_hashes[observed]']
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
