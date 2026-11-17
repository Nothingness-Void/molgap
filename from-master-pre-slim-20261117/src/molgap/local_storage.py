"""Bounded local disposable-cache cleanup; default inspection never removes files.

The inspection command retires only explicitly selected partial source caches.
Builders separately bound their owned cache namespace under an OS lock. Legacy
caches, scratch, package stages and run records remain report-only here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat

from .common.io import _safe_local
from .common.paths import MachinePaths


_CACHE_FILES = frozenset({"source.tar.gz", "SOURCE_COMMIT.txt",
                          "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"})
_CACHE_TEMP_FILES = frozenset(f".{name}.tmp" for name in _CACHE_FILES)
_RETIRED = re.compile(r"\.incomplete-[0-9a-f]{64}-[0-9a-f]{32}\Z")
_PARTIAL = re.compile(r"\.partial-[0-9a-f]{64}-[a-z0-9_]{8}\Z")
_MAX_CANDIDATES = 128
SOURCE_CACHE_NAMESPACE = "bounded-v1"


@contextmanager
def source_cache_session(root):
    """Nonblocking, crash-released lock for the owned cache namespace only.

    Old builders never use this namespace. Busy/unavailable caches fall back to
    direct packaging; no age, PID or lock-file deletion establishes liveness.
    """
    keep = int(os.environ.get("MOLGAP_SOURCE_CACHE_MAX_ENTRIES", "8"))
    if not 0 <= keep <= _MAX_CANDIDATES:
        raise ValueError("MOLGAP_SOURCE_CACHE_MAX_ENTRIES must be between 0 and 128")
    if keep == 0:
        yield None
        return
    cache = Path(root).absolute() / SOURCE_CACHE_NAMESPACE
    _safe_local(cache)
    handle, acquired = None, False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lock = cache / ".lock"
        _safe_local(lock)
        handle = lock.open("a+b")
        if os.name == "nt":
            import msvcrt
            # Windows permits locking a region beyond EOF. Never initialize or
            # append to this byte: another process may already have locked it.
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
    except OSError:
        pass
    try:
        yield cache if acquired else None
    finally:
        if acquired:
            # The caller has copied its archive independently before returning.
            # Cleanup failures must never invalidate a successfully built package.
            try:
                _trim_source_cache(cache, keep)
            except (OSError, ValueError):
                pass
        if handle is not None:
            handle.close()  # Closing releases the OS lock even after exceptions.


def _trim_source_cache(cache, keep):
    """Evict generated cache bytes while every cooperating reader is excluded."""
    children = list(islice(cache.iterdir(), 4097))
    if len(children) > 4096:
        return
    complete, partial = [], []
    for path in children:
        if not (re.fullmatch(r"[0-9a-f]{64}", path.name)
                or _RETIRED.fullmatch(path.name) or _PARTIAL.fullmatch(path.name)):
            continue
        try:
            _safe_local(path)
            if not stat.S_ISDIR(path.lstat().st_mode):
                continue
            # v4_bundle may die before renaming its fixed intermediate files.
            # Only an unpublished .partial directory owns these names.
            allowed = _CACHE_FILES | _CACHE_TEMP_FILES if _PARTIAL.fullmatch(path.name) else _CACHE_FILES
            files = list(islice(path.iterdir(), len(allowed) + 1))
            if any(p.name not in allowed for p in files):
                continue
            for child in files:
                _safe_local(child)
                if not stat.S_ISREG(child.lstat().st_mode):
                    raise ValueError("Cache contains nonregular bytes")
            item = (path, files)
            if re.fullmatch(r"[0-9a-f]{64}", path.name) and {p.name for p in files} == _CACHE_FILES:
                complete.append(item)
            else:
                partial.append(item)
        except (OSError, ValueError):
            continue
    complete.sort(key=lambda item: item[0].stat().st_mtime_ns, reverse=True)
    for path, files in partial + complete[keep:]:
        try:
            _safe_local(path)
            for child in files:
                _safe_local(child)
                child.unlink()
            path.rmdir()
        except (OSError, ValueError):
            continue


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


def _tree_files(root):
    """Bounded inventory with link checks before every descent; never rglob links."""
    _safe_local(root)
    if not root.exists():
        return {}, []
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Disposable root must be a directory")
    files, directories, pending = {}, [], [root]
    count = 0
    while pending:
        directory = pending.pop()
        _safe_local(directory)
        directories.append(directory)
        if len(directories) > 256:
            raise ValueError("Disposable directory inventory exceeds bound")
        for path in directory.iterdir():
            count += 1
            if count > 4352:
                raise ValueError("Disposable object inventory exceeds bound")
            _safe_local(path)
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                from .training_reproducibility import sha256_file
                files[path.relative_to(root).as_posix()] = {
                    "sha256": sha256_file(path), "bytes": info.st_size}
                if len(files) > 4096:
                    raise ValueError("Disposable file inventory exceeds bound")
            else:
                raise ValueError("Disposable tree contains a nonregular object")
    return files, directories


def _remove_verified_files(root, expected, *, whole_tree=False):
    """Idempotent receipt-bound unlink; missing files permit crash recovery."""
    from .experiment_preflight import _relative
    from .training_reproducibility import sha256_file
    root = Path(root).absolute()
    for name in expected:
        _relative(name)
    files, directories = _tree_files(root)
    if whole_tree and set(files) - set(expected):
        raise ValueError("Disposable tree contains unbound files")
    for name, record in expected.items():
        if name in files and files[name] != record:
            raise ValueError("Disposable bytes changed after owner completion")
    # Complete inspection precedes the first mutation. Recheck each path at use.
    for name, record in expected.items():
        path = root / name
        _safe_local(path)
        if path.exists():
            if (not stat.S_ISREG(path.lstat().st_mode)
                    or path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]):
                raise ValueError("Disposable file changed before unlink")
            path.unlink()
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        _safe_local(directory)
        if (whole_tree or directory != root) and not any(directory.iterdir()):
            directory.rmdir()


def retire_failed_preparations(run, spec_identity, package_identity):
    """Retire only unpublished failed copies after the replacement is verified.

    Existing workflow/release reports remain as compact receipts beside each old
    name, then move to completed/ on reentry. A pending retirement receipt is
    retried after interrupted unlink; no process guessing, lock removal, or
    published preparation deletion occurs.
    """
    from .experiment_package import verify_experiment_source_package
    from .common.io import publish_immutable_bytes
    from .training_reproducibility import atomic_json
    run = Path(run).absolute()
    _safe_local(run)
    prepared, failures = run / "prepared", run / "local_preparation_failures"
    _safe_local(failures)
    if not failures.exists():
        return []
    # Bound metadata discovery separately from pending owners: each owner may
    # have both a directory and receipt, plus the unscanned completed namespace.
    # Draining completed receipts gives subsequent calls progress through legacy
    # histories without an unbounded directory scan or payload inventory.
    discovery_bound = 2 * _MAX_CANDIDATES + 1
    children = list(islice(failures.iterdir(), discovery_bound + 1))
    results = []
    if len(children) > discovery_bound:
        results.append({"path": str(failures), "status": "REPORT_ONLY",
                        "reason": "Candidate bound exceeded"})
    names = {child.name.removesuffix(".json") for child in children[:discovery_bound]
             if re.fullmatch(r"prepared-[0-9a-f]{32}(?:\.json)?", child.name)}
    pending = []
    for name in names:
        old, receipt = failures / name, failures / (name + ".json")
        _safe_local(old)
        _safe_local(receipt)
        if not old.exists() and receipt.exists():
            try:
                identity = _identity(receipt)
                payload = receipt.read_bytes()
                retained = json.loads(payload)
                if retained["local_payload_retirement"]["status"] == "REMOVED":
                    try:
                        archived = failures / "completed" / receipt.name
                        _safe_local(archived)
                        publish_immutable_bytes(archived, payload)
                        _safe_local(receipt)
                        if _identity(receipt) != identity:
                            raise ValueError("Completed retirement receipt changed before relocation")
                        receipt.unlink()
                    except (OSError, ValueError) as exc:
                        results.append({"path": str(receipt), "status": "REPORT_ONLY", "reason": str(exc)})
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
        pending.append(name)
    if len(pending) > _MAX_CANDIDATES:
        if len(children) <= discovery_bound:
            results.append({"path": str(failures), "status": "REPORT_ONLY",
                            "reason": "Candidate bound exceeded"})
        pending = sorted(pending)[:_MAX_CANDIDATES]
    if not pending:
        return results  # Completed receipts need no replacement-package rehash.
    manifest = verify_experiment_source_package(prepared / "package")
    report = json.loads((prepared / "workflow_report.json").read_bytes())
    if (manifest["spec_identity"] != spec_identity or manifest["package_identity"] != package_identity
            or report.get("status") != "PREPARED_FOR_PLATFORM"
            or report.get("package_identity") != package_identity):
        raise ValueError("Failed preparation retirement requires its verified replacement")
    replacement, _ = _tree_files(prepared)
    for name in sorted(pending):
        old = failures / name
        result = {"path": str(old), "status": "REPORT_ONLY"}
        try:
            files, _ = _tree_files(old)
            receipt = failures / (old.name + ".json")
            _safe_local(receipt)
            if receipt.exists():
                retained = json.loads(receipt.read_bytes())
                prior = retained["workflow_report"]
                retirement = retained["local_payload_retirement"]
                if retirement["replacement_package_identity"] != package_identity:
                    raise ValueError("Replacement identity changed")
                expected = retirement["files"]
            else:
                report_path = old / "workflow_report.json"
                _safe_local(report_path)
                prior = json.loads(report_path.read_bytes())
                if set(old_path.split("/", 1)[0] for old_path in files) - {
                        "package", "source_dataset", "kernel", "workflow_report.json", "release_report.json"}:
                    raise ValueError("Failed preparation contains unknown payload")
                expected = files
                retained = {"workflow_report": prior,
                    "local_payload_retirement": {"replacement_package_identity": package_identity,
                        "replacement_source_commit": manifest["source_commit"],
                        "files": expected, "status": "PENDING"}}
                if "release_report.json" in files:
                    retained["release_report"] = json.loads((old / "release_report.json").read_bytes())
            # Recheck surviving bytes even on receipt retries: replacement files
            # may have disappeared since an interrupted retirement was approved.
            for name, identity in files.items():
                if name not in {"workflow_report.json", "release_report.json"} and replacement.get(name) != identity:
                    raise ValueError("Failed preparation contains unmatched bytes")
            _safe_local(run / "operations")
            if (prior.get("format") != "molgap-workflow-preparation-v1"
                    or prior.get("spec_identity") != spec_identity
                    or prior.get("status") not in {"ERROR_BEFORE_PUBLICATION", "BLOCKED"}
                    or prior.get("submitted") is not False
                    or prior.get("prospective_published") is not False
                    or prior.get("publication_intent") is not False
                    or (run / "operations").exists()):
                raise ValueError("Failed preparation publication or liveness is uncertain")
            # The existing preparation receipts are durable before the first unlink.
            atomic_json(receipt, retained)
            _remove_verified_files(old, expected, whole_tree=True)
            retained["local_payload_retirement"]["status"] = "REMOVED"
            atomic_json(receipt, retained)
            result["status"] = "REMOVED"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["reason"] = str(exc)
        results.append(result)
    return results


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


_REPORT_ROOTS = ("retrieved", "reconciliation", "acceptance", "recovery",
                 "research", "runs", "diagnostics")
_MIN_DUPLICATE_BYTES = 1 << 20
_NATIVE_PROOF_COPIES = frozenset({"repeat-a.pt", "repeat-b.pt", "resumed.pt",
                                  "resumed-first.pt", "fill-enabled.pt"})


def _category(name):
    lower = name.lower()
    if lower == "last_checkpoint.pt":
        return "resume_state"
    if lower in _NATIVE_PROOF_COPIES:
        return "native_proof_copy"
    if lower.endswith(".pt") and ("selected" in lower or "best" in lower):
        return "selected_model"
    if "prediction" in lower:
        return "predictions"
    if "trace" in lower and lower.endswith(".json"):
        return "trace"
    return "other"


def _walk_files(root):
    """Regular files only; links and reparse points are never followed."""
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                # DirEntry.stat() leaves st_ino/st_dev zero on Windows, which would
                # make every file look like a hardlink alias of every other.
                info = os.lstat(entry.path)
                if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    continue
                if entry.name.endswith(".dedupe-tmp"):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append(entry.path)
                elif stat.S_ISREG(info.st_mode):
                    yield Path(entry.path), info


def storage_report(profile, repo_root):
    """Read-only bytes per run by file-name category, plus exact-SHA duplicates.

    Categories come from file names and only describe the data; they decide
    nothing. Hardlink aliases count once. Only files of equal size are hashed.
    """
    from datetime import datetime
    from .training_reproducibility import sha256_file
    paths = profile if type(profile) is MachinePaths else MachinePaths.from_file(profile)
    paths.outside_checkout(repo_root)
    runs, by_size = {}, {}
    for top in _REPORT_ROOTS:
        base = paths.artifact_root / top
        _safe_local(base)
        if not base.is_dir():
            continue
        for path, info in _walk_files(base):
            parts = path.relative_to(base).parts
            row = runs.setdefault(f"{top}/{parts[0] if len(parts) > 1 else ''}",
                {"bytes": 0, "files": 0, "newest_mtime": 0, "by_category": {}})
            row["bytes"] += info.st_size
            row["files"] += 1
            row["newest_mtime"] = max(row["newest_mtime"], info.st_mtime)
            category = _category(path.name)
            row["by_category"][category] = row["by_category"].get(category, 0) + info.st_size
            if info.st_size >= _MIN_DUPLICATE_BYTES:
                by_size.setdefault(info.st_size, {}).setdefault((info.st_dev, info.st_ino), path)
    groups = []
    for size, aliases in by_size.items():
        if len(aliases) < 2:
            continue
        digests = {}
        for path in aliases.values():
            digests.setdefault(sha256_file(path), []).append(str(path))
        groups.extend({"sha256": digest, "bytes": size, "copies": len(copies),
                       "duplicate_bytes": size * (len(copies) - 1), "paths": sorted(copies)}
                      for digest, copies in digests.items() if len(copies) > 1)
    groups.sort(key=lambda group: group["duplicate_bytes"], reverse=True)
    rows = [{"run": name, "bytes": row["bytes"], "files": row["files"],
             "newest": datetime.fromtimestamp(row["newest_mtime"]).date().isoformat(),
             "by_category": dict(sorted(row["by_category"].items()))}
            for name, row in sorted(runs.items(), key=lambda item: item[1]["bytes"], reverse=True)]
    return {"status": "REPORT", "executed": False,
            "byte_basis": "logical_regular_files; hardlink aliases count once in duplicates",
            "total_bytes": sum(row["bytes"] for row in rows), "runs": rows,
            "duplicate_bytes": sum(group["duplicate_bytes"] for group in groups),
            "duplicate_groups": groups}


def dedupe_artifacts(profile, repo_root, *, execute=False):
    """Report duplicates; broad artifact hardlink mutation is disabled.

    Equal bytes and file names do not establish immutability. Resume writers
    can copy onto existing outputs, mutating every alias or raising SameFileError.
    """
    reason = "Artifact immutability is unverified; hardlink deduplication is disabled"
    if execute:
        return {"status": "BLOCKED", "executed": False, "reason": reason}
    report = storage_report(profile, repo_root)
    return {**report, "status": "DRY_RUN", "reason": reason,
            "reclaimable_bytes": None, "groups": len(report["duplicate_groups"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate", action="append", default=[], type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Read-only bytes per run and exact-SHA duplicate groups")
    parser.add_argument("--dedupe", action="store_true",
                        help="Read-only duplicate report; --execute is blocked until immutability is established")
    args = parser.parse_args(argv)
    try:
        if args.report and args.dedupe:
            raise ValueError("Choose --report or --dedupe")
        if args.dedupe:
            if args.candidate:
                raise ValueError("--dedupe takes no candidates")
            report = dedupe_artifacts(args.profile, args.repo_root, execute=args.execute)
        elif args.report:
            if args.candidate or args.execute:
                raise ValueError("--report is read-only and takes no candidates")
            report = storage_report(args.profile, args.repo_root)
        else:
            report = clean_local_storage(args.profile, args.repo_root,
                                         candidates=args.candidate, execute=args.execute)
    except (OSError, ValueError) as exc:
        report = {"status": "BLOCKED", "executed": False, "reason": str(exc)}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"DRY_RUN", "REMOVED", "REPORT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
