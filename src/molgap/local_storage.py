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
            for relative, identity in files.items():
                if relative not in {"workflow_report.json", "release_report.json"} and replacement.get(relative) != identity:
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
