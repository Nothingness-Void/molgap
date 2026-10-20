"""Synthetic disposable-cache recovery; no actual machine storage is touched."""
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from molgap.local_storage import clean_local_storage, main
from test_machine_paths import profile


def retired(paths, suffix="b" * 32):
    path = paths.artifact_root / "source-packages/runtime-cache" / (".incomplete-" + "a" * 64 + "-" + suffix)
    path.mkdir(parents=True)
    (path / "source.tar.gz").write_bytes(b"reproducible source cache")
    return path


def test_default_dry_run_and_explicit_disposable_removal(tmp_path):
    paths = profile(tmp_path)
    target = retired(paths)
    scratch = paths.scratch_root / "molgap-live"
    scratch.mkdir(parents=True)
    (scratch / "result.pt").write_bytes(b"possible unique state")
    first = clean_local_storage(paths, tmp_path / "code")
    assert first["status"] == "DRY_RUN" and not first["executed"]
    by_path = {row["path"]: row for row in first["candidates"]}
    assert by_path[str(target)]["disposition"] == "DISPOSABLE_RETIRED_CACHE"
    assert by_path[str(scratch)]["reason"] == "SCRATCH_LIVENESS_UNKNOWN"
    assert target.exists() and (scratch / "result.pt").exists()
    with pytest.raises(ValueError, match="explicit"):
        clean_local_storage(paths, tmp_path / "code", execute=True)
    result = clean_local_storage(paths, tmp_path / "code", candidates=[target], execute=True)
    assert result["status"] == "REMOVED" and result["executed"]
    assert not target.exists() and (scratch / "result.pt").exists()


@pytest.mark.parametrize("extra", ["checkpoint.pt", "operations", "nested"])
def test_unknown_payloads_and_nested_objects_block_the_whole_selection(tmp_path, extra):
    paths = profile(tmp_path)
    safe = retired(paths)
    uncertain = retired(paths, "c" * 32)
    if extra == "nested":
        (uncertain / "SOURCE_FILES.json").mkdir()
    else:
        (uncertain / extra).write_bytes(b"preserve")
    result = clean_local_storage(paths, tmp_path / "code", candidates=[safe, uncertain], execute=True)
    assert result["status"] == "BLOCKED" and not result["executed"]
    assert (safe / "source.tar.gz").exists() and uncertain.exists()


@pytest.mark.parametrize("relative", ["runs/question/v1/prepared/package/.package-building",
    "runs/question/v1/prepared/package/.source-stage-old",
    "runs/question/v1/local_preparation_failures/prepared-old",
    "runs/question/v1/operations/journal.json", "source-packages/runtime-cache/.source-cache-old",
    "source-packages/runtime-cache/" + "a" * 64,
    "source-packages/runtime-cache/.incomplete-not-owned"])
def test_locks_stages_journals_failed_preparations_and_live_caches_are_report_only(tmp_path, relative):
    paths = profile(tmp_path)
    target = paths.artifact_root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve")
    result = clean_local_storage(paths, tmp_path / "code", candidates=[target], execute=True)
    assert result["status"] == "BLOCKED" and not result["executed"]
    assert target.read_bytes() == b"preserve"


def test_outside_paths_roots_traversal_and_lookalikes_are_never_removed(tmp_path):
    paths = profile(tmp_path)
    target = retired(paths)
    selections = [paths.data_root, paths.artifact_root, paths.scratch_root,
                  tmp_path / "outside", target / ".." / target.name]
    for selected in selections:
        result = clean_local_storage(paths, tmp_path / "code", candidates=[selected], execute=True)
        assert result["status"] == "BLOCKED" and not result["executed"]
    assert (target / "source.tar.gz").exists()


@pytest.mark.parametrize("link_target", ["candidate", "sidecar", "ancestor"])
def test_symlinks_and_windows_reparse_paths_block_without_following(tmp_path, monkeypatch, link_target):
    paths = profile(tmp_path)
    target = retired(paths)
    blocked = {"candidate": target, "sidecar": target / "source.tar.gz",
               "ancestor": target.parent}[link_target]
    original = Path.lstat

    def reparse(path, *args, **kwargs):
        if path == blocked:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", reparse)
    result = clean_local_storage(paths, tmp_path / "code", candidates=[target], execute=True)
    assert result["status"] == "BLOCKED" and not result["executed"]
    assert (target / "source.tar.gz").read_bytes() == b"reproducible source cache"


def test_candidate_changed_after_planning_blocks_before_first_unlink(tmp_path, monkeypatch):
    from molgap import local_storage
    paths = profile(tmp_path)
    first = retired(paths)
    second = retired(paths, "c" * 32)
    original = local_storage._inspect
    calls = []

    def mutate(path, machine):
        calls.append(path)
        if len(calls) == 3:
            (second / "source.tar.gz").write_bytes(b"changed payload")
        return original(path, machine)

    monkeypatch.setattr(local_storage, "_inspect", mutate)
    result = clean_local_storage(paths, tmp_path / "code", candidates=[first, second], execute=True)
    assert result["status"] == "BLOCKED" and not result["executed"]
    assert (first / "source.tar.gz").exists() and second.exists()


def test_cli_profile_dry_run_and_discovery_bound(tmp_path, capsys):
    paths = profile(tmp_path)
    target = retired(paths)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"format": "molgap-machine-paths-v1", **{
        key: str(getattr(paths, key)) for key in ("python", "data_root", "artifact_root", "scratch_root")}}))
    assert main(["--profile", str(profile_path), "--repo-root", str(tmp_path / "code")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DRY_RUN" and target.exists()
    for index in range(130):
        (target.parent / ("unknown-" + str(index))).mkdir()
    result = clean_local_storage(paths, tmp_path / "code")
    assert result["truncated"] and len(result["candidates"]) == 128
    assert main(["--profile", str(profile_path), "--repo-root", str(tmp_path / "code"), "--execute"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED" and target.exists()
