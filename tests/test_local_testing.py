"""Flow selection must neither duplicate pytest work nor hide its exit status."""
import json
from pathlib import Path

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
