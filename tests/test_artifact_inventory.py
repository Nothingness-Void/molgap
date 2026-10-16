"""Tiny local fixtures only; main delegates execution with E-drive basetemp."""
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from molgap import artifact_inventory as inventory


def git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(["git", "-C", str(root), *args], env=env,
                          check=True, capture_output=True).stdout


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "ended_checkout"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / ".gitignore").write_bytes(b"*.payload\n")
    (root / "source.py").write_bytes(b"initial\n")
    git(root, "add", ".gitignore", "source.py")
    git(root, "commit", "-qm", "Fixture")
    return root


def total_files(report):
    return sum(v["files"] for v in report["totals"].values())


def test_bulk_classification_sizes_and_unknown_custody(repository, monkeypatch):
    root = repository
    secret = b"fixture private payload must not appear in report"
    (root / "retained.payload").write_bytes(secret)
    (root / "uncommitted.txt").write_bytes(b"four")
    (root / "source.py").write_bytes(b"changed source with a different length\n")
    calls = []
    original = inventory._git

    def observed(root, args, **kwargs):
        calls.append(args)
        return original(root, args, **kwargs)

    monkeypatch.setattr(inventory, "_git", observed)
    report = inventory.census_paths([root])
    assert report["complete"]
    assert report["totals"]["tracked"] == {
        "files": 2, "bytes": len(b"*.payload\n") + len(b"changed source with a different length\n")}
    assert report["totals"]["untracked"] == {"files": 1, "bytes": 4}
    assert report["totals"]["ignored"] == {"files": 1, "bytes": len(secret)}
    assert report["paths"][0]["possible_unique_files"] == 2
    assert report["paths"][0]["skipped_git_admin"] == 1
    assert len(calls) == 5
    assert not any(args[0] in {"show", "hash-object", "status"} for args in calls)
    metadata = report["worktrees"][0]
    assert metadata["head"] == git(root, "rev-parse", "HEAD").decode().strip()
    assert metadata["branch"].startswith("refs/heads/")
    assert metadata["dirty"] == "UNKNOWN"
    assert metadata["observed_index_or_stat_changes"] is True
    assert report["custody"] == report["unique_copies"] == "UNKNOWN"
    assert report["deletion_eligibility"] == "NOT_ASSESSED"
    assert report["evidence"] == "INSUFFICIENT_FOR_CLEANUP"
    assert secret.decode() not in json.dumps(report)


def test_subset_never_reports_worktree_clean_or_dumps_contents(repository, monkeypatch):
    selected = repository / "source.py"
    (repository / "outside_selection.txt").write_text("unknown custody", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Census must not open payloads")

    monkeypatch.setattr(Path, "open", forbidden)
    report = inventory.census_paths([selected])
    assert report["complete"]
    assert total_files(report) == 1
    assert report["worktrees"][0]["dirty"] == "UNKNOWN"
    assert report["worktrees"][0]["whole_worktree_dirty"] == "UNKNOWN"


def test_duplicate_and_overlapping_selections(repository):
    sub = repository / "artifacts"
    sub.mkdir()
    (sub / "kept.payload").write_bytes(b"kept")
    report = inventory.census_paths([sub, repository, repository / ".", sub / "kept.payload"])
    assert report["complete"]
    assert total_files(report) == 3
    assert sum(p["duplicate_of"] is not None for p in report["paths"]) == 3
    assert report["totals"]["ignored"] == {"files": 1, "bytes": 4}


@pytest.mark.parametrize("limits", [
    {"max_entries": 2}, {"max_depth": 0}, {"max_git_bytes": 1}])
def test_limit_is_partial_and_dirty_unknown(repository, limits):
    report = inventory.census_paths([repository], **limits)
    assert not report["complete"]
    assert report["evidence"] == "INSUFFICIENT_FOR_CLEANUP"
    assert report["worktrees"][0]["dirty"] == "UNKNOWN"
    assert report["worktrees"][0]["observed_index_or_stat_changes"] == "UNKNOWN"
    if "max_entries" in limits:
        assert report["entries_visited"] <= limits["max_entries"]


@pytest.mark.parametrize("seconds", [True, False, float("nan"), float("inf"),
                                         -float("inf"), 0, -1, "30"])
def test_invalid_seconds_rejected_before_traversal(tmp_path, seconds, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid limits must not traverse")

    monkeypatch.setattr(inventory.os, "scandir", forbidden)
    with pytest.raises(ValueError, match="limits"):
        inventory.census_paths([tmp_path], max_seconds=seconds)


def test_time_limit_prevents_traversal(tmp_path, monkeypatch):
    (tmp_path / "kept.txt").write_bytes(b"kept")
    clock = iter([0, 2])
    monkeypatch.setattr(inventory.time, "monotonic", lambda: next(clock, 2))
    report = inventory.census_paths([tmp_path], max_seconds=1)
    assert not report["complete"]
    assert report["entries_visited"] == 0
    assert report["paths"][0]["issues"] == ["ENTRY_OR_TIME_LIMIT"]


def test_missing_and_nonrepo_paths(tmp_path):
    artifact = tmp_path / "archive.bin"
    artifact.write_bytes(b"retained")
    report = inventory.census_paths([artifact, tmp_path / "missing"])
    assert not report["complete"]
    assert report["totals"]["unknown"] == {"files": 1, "bytes": 8}
    assert report["worktrees"] == []
    assert all(p["possible_unique_files"] >= 0 for p in report["paths"])


def test_native_symlink_root_ancestor_and_child_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "unique.payload").write_bytes(b"do not follow")
    selected = tmp_path / "selected"
    selected.mkdir()
    link = selected / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not permit native symlink creation")
    for root in (selected, link, link / "unique.payload"):
        report = inventory.census_paths([root])
        assert not report["complete"]
        assert total_files(report) == 0
        assert report["worktrees"] == []
    assert inventory.census_paths([selected])["paths"][0]["skipped_links"] == 1


def test_mocked_windows_reparse_is_skipped_without_enumeration(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    junction = selected / "junction"
    junction.mkdir()
    (junction / "unique.payload").write_bytes(b"do not follow")
    real_scandir = os.scandir

    class Entries:
        def __enter__(self):
            self.entries = real_scandir(selected)
            return (SimpleNamespace(path=e.path, stat=lambda **kw: SimpleNamespace(
                st_mode=stat.S_IFDIR, st_file_attributes=0x400)) for e in self.entries)

        def __exit__(self, *args):
            self.entries.close()

    def guarded(path):
        assert Path(path) != junction, "Reparse target enumerated"
        return Entries() if Path(path) == selected else real_scandir(path)

    monkeypatch.setattr(inventory.os, "scandir", guarded)
    report = inventory.census_paths([selected])
    assert not report["complete"]
    assert report["paths"][0]["skipped_links"] == 1
    assert total_files(report) == 0


def test_directory_identity_alias_is_not_counted_twice(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "kept.bin").write_bytes(b"kept")
    (second / "kept.bin").write_bytes(b"kept")
    real_lstat = Path.lstat
    first_info = first.lstat()

    def lstat(path, *args, **kwargs):
        return first_info if path == second else real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    report = inventory.census_paths([first, second])
    assert not report["complete"]
    assert total_files(report) == 1
    assert report["paths"][1]["duplicate_of"] == str(first)


def test_mocked_reparse_root_and_ancestor_rejected_before_git(tmp_path, monkeypatch):
    junction = tmp_path / "junction"
    junction.mkdir()
    real_lstat = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == junction:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return real_lstat(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe selection reached Git or traversal")

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(inventory, "_git", forbidden)
    monkeypatch.setattr(inventory.os, "scandir", forbidden)
    for selected in (junction, junction / "unseen.bin"):
        report = inventory.census_paths([selected])
        assert not report["complete"]
        assert total_files(report) == report["entries_visited"] == 0


def test_git_failure_is_unknown_without_exception_or_stderr_dump(repository, monkeypatch):
    def unavailable(*args, **kwargs):
        raise inventory._GitUnavailable("fixture-secret-error")

    monkeypatch.setattr(inventory, "_git", unavailable)
    report = inventory.census_paths([repository])
    assert not report["complete"]
    assert report["totals"]["unknown"]["files"] == 2
    assert report["worktrees"][0]["dirty"] == "UNKNOWN"
    assert "fixture-secret-error" not in json.dumps(report)


def test_nested_repository_is_excluded(repository):
    nested = repository / "nested"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "unique.bin").write_bytes(b"nested")
    report = inventory.census_paths([repository])
    assert not report["complete"]
    assert total_files(report) == 2
    assert "NESTED_WORKTREE_EXCLUDED" in report["paths"][0]["issues"]


def test_cli_outputs_only_report_and_returns_partial_exit_code(tmp_path, capsys):
    selected = tmp_path / "explicit_artifacts"
    selected.mkdir()
    (selected / "one.bin").write_bytes(b"one")
    assert inventory.main([str(selected), "--max-depth", "0"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["complete"]
    assert report["deletion_eligibility"] == "NOT_ASSESSED"
