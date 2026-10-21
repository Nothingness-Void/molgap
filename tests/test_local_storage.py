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


# Reuse the source-only package fixture; no scientific payload or remote work.
from test_experiment_package import repo, package
from test_experiment_spec import payload


@pytest.fixture
def failed_preparation(tmp_path, package):
    import shutil
    from molgap.experiment_package import verify_experiment_source_package
    run = tmp_path / "run"
    prepared = run / "prepared"
    prepared.mkdir(parents=True)
    shutil.copytree(package, prepared / "package")
    manifest = verify_experiment_source_package(prepared / "package")
    (prepared / "workflow_report.json").write_text(json.dumps({
        "status": "PREPARED_FOR_PLATFORM", "package_identity": manifest["package_identity"]}))
    old = run / "local_preparation_failures" / ("prepared-" + "a" * 32)
    old.mkdir(parents=True)
    shutil.copytree(package, old / "package")
    report = {"format": "molgap-workflow-preparation-v1", "spec_identity": manifest["spec_identity"],
        "status": "ERROR_BEFORE_PUBLICATION", "submitted": False,
        "prospective_published": False, "publication_intent": False, "error": "synthetic preflight error"}
    (old / "workflow_report.json").write_text(json.dumps(report))
    def retire():
        from molgap.local_storage import retire_failed_preparations
        return retire_failed_preparations(run, manifest["spec_identity"], manifest["package_identity"])
    return SimpleNamespace(run=run, old=old, manifest=manifest, retire=retire, report=report)


def test_failed_preparation_retirement_preserves_compact_original_receipts(failed_preparation):
    case = failed_preparation
    assert case.retire()[0]["status"] == "REMOVED"
    assert not case.old.exists()
    receipt = json.loads(case.old.with_name(case.old.name + ".json").read_bytes())
    assert receipt["workflow_report"] == case.report
    assert receipt["local_payload_retirement"]["status"] == "REMOVED"
    assert (case.run / "prepared/package/source.tar.gz").is_file()


@pytest.mark.parametrize("change", ["operations", "unknown", "published", "newbytes", "link"])
def test_failed_preparation_uncertainty_and_unique_bytes_are_preserved(failed_preparation, change, monkeypatch):
    case = failed_preparation
    if change == "operations":
        (case.run / "operations").mkdir()
    elif change == "unknown":
        (case.old / "unique_checkpoint.pt").write_bytes(b"unique")
    elif change == "published":
        record = dict(case.report, publication_intent=True)
        (case.old / "workflow_report.json").write_text(json.dumps(record))
    elif change == "newbytes":
        (case.old / "package/source.tar.gz").write_bytes(b"unmatched source")
    else:
        original = Path.lstat
        def reparse(path, *args, **kwargs):
            if path == case.old / "package":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", reparse)
    assert case.retire()[0]["status"] == "REPORT_ONLY"
    assert (case.old / "package/source.tar.gz").exists()


def test_failed_preparation_interrupted_unlink_recovers_from_receipt(failed_preparation, monkeypatch):
    case = failed_preparation
    original = Path.unlink
    crashed = []
    def unlink(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.is_relative_to(case.old) and not crashed:
            crashed.append(True)
            raise OSError("synthetic power loss after unlink")
        return result
    monkeypatch.setattr(Path, "unlink", unlink)
    assert case.retire()[0]["status"] == "REPORT_ONLY"
    receipt = json.loads(case.old.with_name(case.old.name + ".json").read_bytes())
    assert receipt["local_payload_retirement"]["status"] == "PENDING"
    assert case.retire()[0]["status"] == "REMOVED" and not case.old.exists()
