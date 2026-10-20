"""Bounded local disposable-cache cleanup; default inspection never removes files.

Only atomically retired partial source caches have a disposable disposition.
Scratch, package locks/stages, complete caches and run records have no reliable
local liveness proof here and remain report-only, regardless of their age.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import re
import stat

from .common.io import _safe_local
from .common.paths import MachinePaths


_CACHE_FILES = frozenset({"source.tar.gz", "SOURCE_COMMIT.txt",
                          "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"})
_RETIRED = re.compile(r"\.incomplete-[0-9a-f]{64}-[0-9a-f]{32}\Z")
_MAX_CANDIDATES = 128


def _identity(path):
    info = path.lstat()
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _inspect(path, paths):
    cache = paths.artifact_root / "source-packages/runtime-cache"
    item = {"path": str(path), "disposition": "REPORT_ONLY", "removed": False,
            "logical_bytes": None}
    if not (path.parent in {cache, paths.scratch_root}
            or path.is_relative_to(paths.artifact_root / "runs")):
        return {**item, "reason": "OUTSIDE_DISPOSABLE_NAMESPACE"}, None
    try:
        _safe_local(path)
        if not path.exists():
            return {**item, "reason": "MISSING"}, None
        if path.parent == paths.scratch_root:
            return {**item, "reason": "SCRATCH_LIVENESS_UNKNOWN"}, None
        if path.parent != cache:
            return {**item, "reason": "RUN_EVIDENCE_OR_PACKAGE_LIVENESS_UNKNOWN"}, None
        if _RETIRED.fullmatch(path.name) is None:
            return {**item, "reason": "CACHE_LIVENESS_UNKNOWN"}, None
        if not stat.S_ISDIR(path.lstat().st_mode):
            return {**item, "reason": "RETIRED_CACHE_NOT_DIRECTORY"}, None
        # A retired object is no longer a builder's cache target. Restrict the
        # payload to its four generated sidecars; unknown files may be evidence.
        before = _identity(path)
        children = list(islice(path.iterdir(), len(_CACHE_FILES) + 1))
        if any(child.name not in _CACHE_FILES for child in children):
            return {**item, "reason": "UNEXPECTED_RETIRED_CACHE_CONTENT"}, None
        snapshot = {}
        for child in children:
            _safe_local(child)
            identity = _identity(child)
            if not stat.S_ISREG(identity[2]):
                return {**item, "reason": "NONREGULAR_RETIRED_CACHE_CONTENT"}, None
            snapshot[child.name] = identity
        if _identity(path) != before:
            return {**item, "reason": "CHANGED_DURING_INSPECTION"}, None
        return {**item, "disposition": "DISPOSABLE_RETIRED_CACHE",
                "reason": "ATOMICALLY_RETIRED_GENERATED_SOURCE_CACHE",
                "logical_bytes": sum(value[3] for value in snapshot.values())}, (before, snapshot)
    except (OSError, ValueError) as exc:
        return {**item, "reason": "UNSAFE_OR_UNREADABLE_PATH",
                "detail": str(exc)}, None


def clean_local_storage(profile, repo_root, *, candidates=(), execute=False):
    """List bounded candidates, or remove explicitly named retired partial caches.

    There is no recursive removal, age policy, process guessing, lock theft or
    run traversal. Explicit execution fails closed if any selection is ineligible.
    Logical bytes count regular files by pathname, not allocated disk savings.
    """
    paths = profile if type(profile) is MachinePaths else MachinePaths.from_file(profile)
    paths.outside_checkout(repo_root)
    selected = [Path(p).absolute() for p in islice(candidates, _MAX_CANDIDATES + 1)]
    if len(selected) > _MAX_CANDIDATES or len(set(selected)) != len(selected):
        raise ValueError("Select at most 128 unique candidate paths")
    if execute and not selected:
        raise ValueError("Cleanup execution requires explicit --candidate paths")
    truncated = False
    if not selected:
        for root in (paths.scratch_root, paths.artifact_root / "source-packages/runtime-cache"):
            _safe_local(root)
            if root.exists():
                children = list(islice(root.iterdir(), _MAX_CANDIDATES + 1))
                room = _MAX_CANDIDATES - len(selected)
                selected.extend(children[:room])
                truncated |= len(children) > room
    checked = [_inspect(path, paths) for path in selected]
    report = {"status": "DRY_RUN", "executed": False, "truncated": truncated,
              "byte_basis": "logical_regular_files; hardlinks count by path",
              "candidates": [item for item, _ in checked]}
    if not execute:
        return report
    if any(snapshot is None for _, snapshot in checked):
        return {**report, "status": "BLOCKED"}
    # Revalidate every selected target before the first unlink, as well as each
    # target immediately before its own removal.
    if any(_inspect(path, paths)[1] != snapshot
           for path, (_, snapshot) in zip(selected, checked)):
        return {**report, "status": "BLOCKED", "reason": "CANDIDATE_CHANGED"}
    report["status"] = "REMOVED"
    for path, (item, snapshot) in zip(selected, checked):
        try:
            _, current = _inspect(path, paths)
            if current != snapshot:
                raise ValueError("Candidate changed after inspection")
            # No rmtree: never traverse nested directories or follow links.
            for name, identity in snapshot[1].items():
                child = path / name
                _safe_local(child)
                if _identity(child) != identity:
                    raise ValueError("Cache sidecar changed before removal")
                child.unlink()
                report["executed"] = True
            _safe_local(path)
            path.rmdir()
            item["removed"] = True
            report["executed"] = True
        except (OSError, ValueError) as exc:
            item["reason"] = "REMOVAL_STOPPED"
            item["detail"] = str(exc)
            report["status"] = "STOPPED"
            break
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = clean_local_storage(args.profile, args.repo_root,
                                     candidates=args.candidate, execute=args.execute)
    except (OSError, ValueError) as exc:
        report = {"status": "BLOCKED", "executed": False, "reason": str(exc)}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"DRY_RUN", "REMOVED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
