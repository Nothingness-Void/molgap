"""Bounded, read-only census of explicitly selected local paths.

Counts are logical regular-file bytes, not allocated disk space or proof of
recoverability. No payload is opened or hashed. Custody, unique copies, process
dependencies and deletion eligibility cannot be established by this census.
Callers select ended worktrees/artifacts; this module does not discover them.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import threading
import time

from .common.io import _safe_local


class _GitUnavailable(Exception):
    pass


def _git(root, args, *, deadline, max_bytes, stdin=None):
    """Bound both output memory and wall time; never expose Git diagnostics."""
    if time.monotonic() >= deadline:
        raise _GitUnavailable
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    # check-ignore consumes literal pathname records, not pathspecs; Git rejects
    # pathspec magic (including the global literal flag) for this command.
    literal = [] if args[0] == "check-ignore" else ["--literal-pathspecs"]
    command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
               "-c", "core.untrackedCache=false", *literal,
               "-C", str(root), *args]
    output = []

    def feed():
        try:
            process.stdin.write(stdin)
        except (OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def receive():
        try:
            output.append(process.stdout.read(max_bytes + 1))
        except (OSError, ValueError):
            pass

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None
                                   else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, env=env)
    except OSError:
        raise _GitUnavailable from None
    writer = threading.Thread(target=feed, daemon=True) if stdin is not None else None
    if writer is not None:
        writer.start()
    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    reader.join(max(0, min(10, deadline - time.monotonic())))
    if reader.is_alive() or not output or len(output[0]) > max_bytes:
        process.kill()
        process.wait()
        reader.join()
        if writer is not None:
            writer.join()
        process.stdout.close()
        raise _GitUnavailable
    try:
        code = process.wait(timeout=max(.001, min(10, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise _GitUnavailable from None
    finally:
        if writer is not None:
            writer.join()
        process.stdout.close()
    if code not in ({0, 1} if args[0] == "check-ignore" else {0}):
        raise _GitUnavailable
    return output[0]


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400)


def _key(path):
    return os.path.normcase(str(path))


def _repo_for(path):
    """Ascend only for a Git marker, without following an unsafe ancestor."""
    for directory in (path.parent, *path.parent.parents) if path.is_file() else (path, *path.parents):
        marker = directory / ".git"
        try:
            info = marker.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info):
            raise ValueError("Unsafe Git marker")
        _safe_local(directory)
        return directory
    return None


def census_paths(paths, *, max_entries=100_000, max_depth=32,
                 max_seconds=30.0, max_git_bytes=4_000_000):
    """Inventory only explicit paths with global entry/time and per-tree depth caps.

    Symlink/reparse roots or ancestors are rejected before Git or traversal.
    Repeated/overlapping selections and directory identity aliases are skipped.
    Git dirty checks use index/stat changes for observed safe regular files;
    they can overreport touched files because contents are never compared. Batches
    it never declares an entire worktree clean. Git administrative directories
    are excluded. No running-process or ended-task assertion is inferred.
    """
    if (type(max_entries) is not int or max_entries <= 0
            or type(max_depth) is not int or max_depth < 0
            or type(max_seconds) not in (float, int)
            or not 0 < max_seconds <= 3600
            or not math.isfinite(max_seconds)
            or type(max_git_bytes) is not int or max_git_bytes <= 0):
        raise ValueError("Invalid census limits")
    selected = [Path(p).absolute() for p in islice(paths, 129)]
    if not selected or len(selected) > 128:
        raise ValueError("Select between one and 128 local paths")
    deadline = time.monotonic() + max_seconds
    report = {
        "format": "molgap-artifact-census-v1", "complete": True,
        "evidence": "INSUFFICIENT_FOR_CLEANUP", "custody": "UNKNOWN",
        "unique_copies": "UNKNOWN", "process_dependencies": "UNKNOWN",
        "deletion_eligibility": "NOT_ASSESSED",
        "byte_basis": "logical_regular_files; hardlinks count by path",
        "limits": dict(max_entries=max_entries, max_depth=max_depth,
                       max_seconds=max_seconds, max_git_bytes=max_git_bytes),
        "entries_visited": 0, "paths": [], "worktrees": [],
    }
    seen_directories = {}
    accepted = []
    repositories = {}

    def issue(item, reason):
        item["complete"] = False
        report["complete"] = False
        if reason not in item["issues"]:
            item["issues"].append(reason)

    # Parents first makes overlap independent of the caller's ordering.
    for path in sorted(selected, key=lambda p: (len(p.parts), _key(p))):
        item = {"path": str(path), "complete": True, "issues": [],
                "counts": {k: {"files": 0, "bytes": 0} for k in
                           ("tracked", "untracked", "ignored", "unknown")},
                "skipped_links": 0, "skipped_special": 0,
                "skipped_git_admin": 0, "duplicate_of": None,
                "possible_unique_files": 0}
        report["paths"].append(item)
        try:
            _safe_local(path)
            if path == Path(path.anchor):
                raise ValueError("Filesystem root")
            root_info = path.lstat()
            if any(part.casefold() == ".git" for part in path.parts):
                raise ValueError("Git administration")
        except (OSError, ValueError):
            issue(item, "UNSAFE_OR_UNAVAILABLE_SELECTION")
            continue
        duplicate = next((p for p in accepted if _key(path) == _key(p)
                          or p in path.parents), None)
        if duplicate is not None:
            item["duplicate_of"] = str(duplicate)
            continue
        accepted.append(path)
        files = []
        stack = [(path, 0, root_info)]
        while stack:
            if report["entries_visited"] >= max_entries or time.monotonic() >= deadline:
                issue(item, "ENTRY_OR_TIME_LIMIT")
                break
            current, depth, info = stack.pop()
            report["entries_visited"] += 1
            if _is_link(info):
                item["skipped_links"] += 1
                issue(item, "LINK_OR_REPARSE_EXCLUDED")
                continue
            if current.name.casefold() == ".git":
                item["skipped_git_admin"] += 1
                continue
            if stat.S_ISREG(info.st_mode):
                files.append((current, info.st_size))
                continue
            if not stat.S_ISDIR(info.st_mode):
                item["skipped_special"] += 1
                issue(item, "SPECIAL_FILE_EXCLUDED")
                continue
            identity = (info.st_dev, info.st_ino) if info.st_ino else _key(current)
            if identity in seen_directories:
                issue(item, "DIRECTORY_ALIAS_EXCLUDED")
                if current == path:
                    item["duplicate_of"] = seen_directories[identity]
                continue
            seen_directories[identity] = str(current)
            if depth >= max_depth:
                issue(item, "DEPTH_LIMIT")
                continue
            try:
                _safe_local(current)
                if current != path:
                    try:
                        (current / ".git").lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        issue(item, "NESTED_WORKTREE_EXCLUDED")
                        continue
                # Iteration is capped before stat or buffering each child.
                with os.scandir(current) as entries:
                    for entry in entries:
                        if (report["entries_visited"] + len(stack) >= max_entries
                                or time.monotonic() >= deadline):
                            issue(item, "ENTRY_OR_TIME_LIMIT")
                            break
                        stack.append((Path(entry.path), depth + 1,
                                      entry.stat(follow_symlinks=False)))
            except (OSError, ValueError):
                issue(item, "PATH_UNAVAILABLE_OR_CHANGED")
        try:
            repo = _repo_for(path)
        except (OSError, ValueError):
            repo = None
            issue(item, "GIT_METADATA_UNAVAILABLE")
        tracked, ignored = set(), set()
        relative = []
        git_ok = False
        if repo is not None:
            item["worktree"] = str(repo)
            metadata = repositories.get(_key(repo))
            if metadata is None:
                metadata = {"path": str(repo), "head": None, "branch": None,
                            "dirty": "UNKNOWN", "dirty_scope": "observed_regular_files_only",
                            "observed_index_or_stat_changes": None,
                            "dirty_basis": "index/stat changes; contents not compared",
                            "status_complete": True,
                            "whole_worktree_dirty": "UNKNOWN"}
                repositories[_key(repo)] = metadata
                report["worktrees"].append(metadata)
                try:
                    values = _git(repo, ["rev-parse", "HEAD", "--symbolic-full-name", "HEAD"],
                                  deadline=deadline, max_bytes=max_git_bytes).splitlines()
                    metadata["head"] = os.fsdecode(values[0])
                    metadata["branch"] = os.fsdecode(values[1]) if len(values) > 1 else None
                except _GitUnavailable:
                    issue(item, "GIT_METADATA_UNAVAILABLE")
            try:
                relative = [os.fsencode(p.relative_to(repo).as_posix()) for p, _ in files]
                tracked = set(_git(repo, ["ls-files", "--cached", "-z", "--",
                                         path.relative_to(repo).as_posix() or "."],
                                   deadline=deadline, max_bytes=max_git_bytes).split(b"\0"))
                if relative:
                    ignored = set(_git(repo, ["check-ignore", "--stdin", "-z"],
                                       deadline=deadline, max_bytes=max_git_bytes,
                                       stdin=b"\0".join(relative) + b"\0").split(b"\0"))
                # Raw diffs avoid content hashing/rename detection and index refresh.
                # Bound argv for Windows, never one process per file.
                dirty = False
                batch, length = [], 0
                for name in [*relative, None]:
                    text = os.fsdecode(name) if name is not None else None
                    if batch and (text is None or length + len(text) > 12_000):
                        for command in (["diff-files", "--raw", "-z", "--no-renames",
                                         "--no-ext-diff", "--no-textconv", "--ignore-submodules=all"],
                                        ["diff-index", "--cached", "--raw", "-z", "--no-renames",
                                         "--no-ext-diff", "--no-textconv", "HEAD"]):
                            dirty |= bool(_git(repo, [*command, "--", *batch],
                                               deadline=deadline, max_bytes=max_git_bytes))
                        batch, length = [], 0
                    if text is not None:
                        # A path may have changed since its directory was observed.
                        _safe_local(repo / text)
                        batch.append(text)
                        length += len(text) + 1
                dirty |= any(name not in tracked and name not in ignored for name in relative)
                if files and metadata["status_complete"]:
                    metadata["observed_index_or_stat_changes"] = bool(
                        metadata["observed_index_or_stat_changes"] or dirty)
                git_ok = True
            except (_GitUnavailable, OSError, ValueError):
                metadata["status_complete"] = False
                metadata["observed_index_or_stat_changes"] = "UNKNOWN"
                issue(item, "GIT_CLASSIFICATION_INCOMPLETE")
            if not item["complete"]:
                metadata["status_complete"] = False
                metadata["observed_index_or_stat_changes"] = "UNKNOWN"
        for index, (_, size) in enumerate(files):
            category = "unknown"
            if git_ok:
                name = relative[index]
                category = "tracked" if name in tracked else "ignored" if name in ignored else "untracked"
            item["counts"][category]["files"] += 1
            item["counts"][category]["bytes"] += size
        if repo is None and files:
            item["issues"].append("NO_GIT_CLASSIFICATION")
        item["possible_unique_files"] = sum(item["counts"][k]["files"]
                                            for k in ("untracked", "ignored", "unknown"))
    report["totals"] = {k: {field: sum(p["counts"][k][field] for p in report["paths"])
                            for field in ("files", "bytes")}
                        for k in ("tracked", "untracked", "ignored", "unknown")}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Explicit ended-worktree/artifact paths")
    parser.add_argument("--max-entries", type=int, default=100_000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-git-bytes", type=int, default=4_000_000)
    args = parser.parse_args(argv)
    try:
        result = census_paths(args.paths, max_entries=args.max_entries,
                              max_depth=args.max_depth, max_seconds=args.max_seconds,
                              max_git_bytes=args.max_git_bytes)
    except ValueError:
        parser.error("Invalid selections or census limits")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
