"""Read-only checks for active Markdown pointers and document growth.

The repository already has layout tests for a few fixed entrypoint rules.  This
module keeps the reusable part of that check in the installed package so a new
agent can audit a checkout without importing test code.  It deliberately
audits tracked Markdown by default and skips frozen/generated trees; callers
can pass an explicit document list when reviewing a small fixture.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit


ACTIVE_ROOTS = frozenset({
    "data",
    "docs",
    "experiments",
    "models",
    "platforms",
    "production",
    "research_memory",
    "src",
    "tests",
})
REPO_SKILL_PREFIX = ".agents/skills/molgap-experiment-reuse/"
ROOT_DOCUMENTS = frozenset({
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CLAUDE.md",
    "CURRENT_STATE.md",
    "NAMING.md",
    "README.md",
    "ROADMAP.md",
    "TRACKS.md",
})
# These trees contain frozen evidence or generated payloads.  Their historical
# pointers are not part of the active navigation graph.  Use --include-frozen
# when auditing one of them intentionally.
FROZEN_PARTS = frozenset({
    "_closed",
    "_records",
    "_retired",
    "_staging",
    "archive",
    "bundle",
    "history",
    "packages",
})
LOCAL_SUFFIXES = frozenset({
    ".csv",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".json",
    ".markdown",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".py",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
})
# Stable navigation budgets.  Conditional indexes and dated evidence are
# intentionally absent: their contents may grow while the entry protocols stay
# short and point to them.  Pass these limits explicitly to the library or use
# --check-entrypoint-budgets in the read-only CLI.
ENTRYPOINT_LINE_LIMITS = {
    "AGENTS.md": 170,
    "CURRENT_STATE.md": 120,
    "ROADMAP.md": 120,
    "ARCHITECTURE.md": 140,
    "experiments/README.md": 90,
    "production/README.md": 90,
    "platforms/README.md": 120,
    "docs/operations/EXPERIMENT_QUICKSTART.md": 160,
}

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LINK = re.compile(r"(?<!!)(?:\[[^\]\n]+\])\((<[^>\n]+>|[^)\n]+)\)")
_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_HTML_ANCHOR = re.compile(
    r"<(?:a|span)\b[^>]*\b(?:id|name)\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentationIssue:
    """One actionable documentation problem."""

    document: str
    line: int
    kind: str
    pointer: str
    message: str


@dataclass(frozen=True)
class DocumentationAudit:
    """Machine-readable result of :func:`audit_repository`."""

    repo_root: str
    documents_scanned: int
    pointers_checked: int
    anchors_checked: int
    ignored_pointers: int
    line_counts: dict[str, int]
    line_limits: dict[str, int]
    skipped_documents: tuple[str, ...]
    issues: tuple[DocumentationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["skipped_documents"] = list(self.skipped_documents)
        result["issues"] = [asdict(issue) for issue in self.issues]
        result["ok"] = self.ok
        return result


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_frozen(relative: str) -> bool:
    parts = Path(relative).parts
    if relative.startswith("docs/archive/") or relative == "docs/archive":
        return True
    return bool(FROZEN_PARTS.intersection(parts))


def _is_active_document(relative: str, *, include_frozen: bool) -> bool:
    if include_frozen or not _is_frozen(relative):
        first = Path(relative).parts[0] if Path(relative).parts else ""
        return (
            relative in ROOT_DOCUMENTS
            or first in ACTIVE_ROOTS
            or relative.startswith(REPO_SKILL_PREFIX)
        )
    return False


def tracked_markdown_files(repo_root: str | Path, *, include_frozen: bool = False) -> tuple[Path, ...]:
    """Return tracked active Markdown files without reading ignored payloads."""

    root = Path(repo_root).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.md", "*.markdown"],
            check=True,
            capture_output=True,
            text=False,
        )
        names = [item for item in completed.stdout.decode("utf-8").split("\0") if item]
    except (OSError, subprocess.CalledProcessError):
        # A temporary fixture need not be a Git checkout.  The explicit active
        # root filter still prevents a broad scan of caches and virtualenvs.
        names = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".markdown"}:
                try:
                    names.append(_relative(root, path))
                except ValueError:
                    continue
    paths: list[Path] = []
    for name in sorted(set(names)):
        relative = Path(name).as_posix()
        if not _is_active_document(relative, include_frozen=include_frozen):
            continue
        path = root / Path(relative)
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return tuple(paths)


def _normalise_heading(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    # GitHub-style slugs retain letters, numbers, underscores and hyphens.
    # GitHub removes punctuation such as ``/`` and ``:`` rather than turning
    # it into another word separator; whitespace alone becomes ``-``.
    value = "".join(
        char if (char.isalnum() or char in "-_") else (" " if char.isspace() else "")
        for char in value
    )
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", value)).strip("-")


def markdown_anchors(text: str) -> frozenset[str]:
    """Return explicit and GitHub-style generated anchors for Markdown text."""

    anchors: set[str] = set()
    slug_counts: dict[str, int] = {}
    fenced = False
    fence_char = ""
    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not fenced:
                fenced, fence_char = True, marker
            elif marker == fence_char:
                fenced = False
            continue
        if fenced:
            continue
        heading = _HEADING.match(line)
        if heading:
            base = _normalise_heading(heading.group(2))
            if base:
                index = slug_counts.get(base, 0)
                slug_counts[base] = index + 1
                anchors.add(base if index == 0 else f"{base}-{index}")
        anchors.update(_HTML_ANCHOR.findall(line))
    return frozenset(anchors)


def _in_inline_code(line: str, index: int) -> bool:
    # Count unescaped backticks before the match.  Fenced lines are filtered by
    # the caller, and this intentionally handles the common single-backtick
    # pointer form used in the repository.
    escaped = False
    count = 0
    for char in line[:index]:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "`" and not escaped:
            count += 1
        escaped = False
    return count % 2 == 1


def _split_destination(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    # A Markdown title follows the destination after whitespace.  Paths with
    # spaces should use angle brackets; treating the first token as the path is
    # safer than silently resolving the title as part of a filename.
    return raw.split()[0] if raw.split() else ""


def _looks_like_pointer(value: str) -> bool:
    value = value.strip()
    if not value or value.startswith(("#", "http://", "https://", "mailto:", "ftp://")):
        return False
    parsed = urlsplit(value)
    if parsed.scheme:
        return False
    if any(token in value for token in ("*", "<", ">", "{", "}")):
        return False
    normalized = value.replace("\\", "/").split("#", 1)[0]
    first = normalized.split("/", 1)[0]
    return (
        normalized.startswith(("./", "../"))
        or first in ACTIVE_ROOTS
        or normalized in ROOT_DOCUMENTS
        or normalized.lower().endswith(tuple(LOCAL_SUFFIXES))
    )


def _resolve_local(
    root: Path,
    document: Path,
    pointer: str,
    *,
    kind: str,
) -> tuple[Path | None, str | None]:
    """Resolve a local Markdown destination, reporting unsafe traversal."""

    raw_path = unquote(pointer.split("#", 1)[0]).strip()
    if not raw_path:
        return document, None
    normalized = raw_path.replace("\\", "/")
    if kind == "code_pointer":
        normalized = re.sub(r":(?:\d+|[A-Za-z_]\w*)$", "", normalized)
    first = normalized.split("/", 1)[0]
    base = root if normalized in ROOT_DOCUMENTS or first in ACTIVE_ROOTS else document.parent
    candidate = (base / Path(normalized)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "pointer escapes repository root"
    if not candidate.exists() and kind == "code_pointer":
        # ARCHITECTURE.md uses bare package filenames in its module map.  A
        # unique source-package convention lets us resolve those pointers
        # without treating prose such as ``decision.md`` as a broken path.
        package_candidates = [root / "src" / "molgap" / Path(normalized)]
        if "/" not in normalized:
            package_candidates.append(root / "tests" / Path(normalized))
        if normalized.startswith(("molgap/", "research_memory/")):
            package_candidates.append(root / "src" / Path(normalized))
        for package_candidate in package_candidates:
            if package_candidate.is_file():
                candidate = package_candidate.resolve()
                break
    if not candidate.exists():
        return candidate, "target does not exist"
    return candidate, None


def _iter_pointers(text: str) -> Iterable[tuple[int, str, str]]:
    """Yield ``(line, pointer, kind)`` for links and probable code pointers."""

    fenced = False
    fence_char = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not fenced:
                fenced, fence_char = True, marker
            elif marker == fence_char:
                fenced = False
            continue
        if fenced:
            continue
        for match in _LINK.finditer(line):
            if not _in_inline_code(line, match.start()):
                pointer = _split_destination(match.group(1))
                if pointer:
                    yield line_number, pointer, "markdown_link"
        for match in _CODE.finditer(line):
            pointer = match.group(1).strip()
            # Inline code containing spaces is normally a shell command or a
            # prose example, not a single repository path.  Markdown links
            # remain the strict form for paths with an explicit destination.
            if not any(char.isspace() for char in pointer) and _looks_like_pointer(pointer):
                yield line_number, pointer, "code_pointer"


def _is_ignored_asset(pointer: str, *, kind: str = "markdown_link") -> bool:
    """Return whether a missing destination is an ignored/generated payload."""

    normalized = unquote(pointer.split("#", 1)[0]).replace("\\", "/").lower()
    parts = tuple(part for part in normalized.split("/") if part)
    if not parts:
        return False
    if any(token in normalized for token in ("$", "...", "phasen")):
        return True
    if parts[0] in {"data", "models"}:
        return True
    if len(parts) >= 2 and parts[:2] in {
        ("platforms", "_records"),
        ("platforms", "_staging"),
    }:
        return True
    if "history" in parts or "_closed" in parts or "_retired" in parts:
        return True
    if Path(parts[-1]).name in {
        "source_commit.txt",
        "source_archive_sha256.txt",
        "source_tree_sha256.txt",
    }:
        return True
    if kind == "code_pointer" and Path(parts[-1]).suffix in {
        ".json", ".log", ".npz", ".pt", ".pth", ".safetensors",
    }:
        return True
    if Path(parts[-1]).suffix == ".json" and any(
        part in {"cache", "gpu", "cpu", "output", "paths", "results", "rml_plan", "rml_finalized", "verification_recovery"}
        for part in parts
    ):
        return True
    return Path(parts[-1]).suffix in {
        ".bin", ".ckpt", ".csv", ".jpeg", ".jpg", ".log", ".parquet",
        ".png", ".pt", ".pth", ".safetensors",
    }


def _parse_limits(values: Sequence[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        relative, separator, raw_limit = value.partition("=")
        if not separator or not relative or not raw_limit.isdigit() or int(raw_limit) < 1:
            raise ValueError(f"line limit must be PATH=POSITIVE_INTEGER: {value}")
        limits[Path(relative.replace("\\", "/")).as_posix()] = int(raw_limit)
    return limits


def audit_repository(
    repo_root: str | Path,
    *,
    documents: Sequence[str | Path] | None = None,
    include_frozen: bool = False,
    line_limits: Mapping[str, int] | None = None,
) -> DocumentationAudit:
    """Audit local Markdown links, anchors, code pointers and optional limits."""

    root = Path(repo_root).resolve()
    if documents is None:
        paths = tracked_markdown_files(root, include_frozen=include_frozen)
        skipped: list[str] = []
    else:
        paths = []
        skipped = []
        for supplied in documents:
            candidate = Path(supplied)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()
            try:
                relative = _relative(root, candidate)
            except ValueError:
                skipped.append(str(supplied))
                continue
            if not include_frozen and _is_frozen(relative):
                skipped.append(relative)
                continue
            if candidate.is_file() and not candidate.is_symlink():
                paths.append(candidate)
            else:
                skipped.append(relative)

    limits = {
        Path(key.replace("\\", "/")).as_posix(): int(value)
        for key, value in (line_limits or {}).items()
    }
    issues: list[DocumentationIssue] = []
    line_counts: dict[str, int] = {}
    pointers_checked = 0
    anchors_checked = 0
    ignored_pointers = 0
    for document in sorted(set(paths)):
        relative = _relative(root, document)
        text = document.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        line_counts[relative] = len(lines)
        limit = limits.get(relative)
        if limit is not None and len(lines) > limit:
            issues.append(DocumentationIssue(
                relative, 1, "line_limit", relative,
                f"document has {len(lines)} lines; limit is {limit}",
            ))
        cache: dict[Path, frozenset[str]] = {document: markdown_anchors(text)}
        for line_number, pointer, kind in _iter_pointers(text):
            if not _looks_like_pointer(pointer):
                continue
            pointers_checked += 1
            target, error = _resolve_local(root, document, pointer, kind=kind)
            if error:
                # Bare Markdown/JSON names in prose are often role labels or
                # placeholders (for example ``decision.md`` in AGENTS.md).
                # Linked destinations remain strict; only unlinked prose is
                # ignored when no direct or package-relative file exists.
                bare_code_pointer = (
                    kind == "code_pointer"
                    and "/" not in pointer.replace("\\", "/")
                )
                if bare_code_pointer and error == "target does not exist":
                    continue
                if error == "target does not exist" and _is_ignored_asset(pointer, kind=kind):
                    ignored_pointers += 1
                    continue
                issues.append(DocumentationIssue(relative, line_number, "missing_path" if error == "target does not exist" else "unsafe_path", pointer, error))
                continue
            assert target is not None
            fragment = unquote(pointer.split("#", 1)[1]).strip() if "#" in pointer else ""
            if fragment and target.is_file() and target.suffix.lower() in {".md", ".markdown"}:
                if target not in cache:
                    cache[target] = markdown_anchors(target.read_text(encoding="utf-8", errors="replace"))
                anchors_checked += 1
                if fragment not in cache[target]:
                    issues.append(DocumentationIssue(relative, line_number, "missing_anchor", pointer, f"anchor not found in {_relative(root, target)}"))
    return DocumentationAudit(
        repo_root=str(root),
        documents_scanned=len(paths),
        pointers_checked=pointers_checked,
        anchors_checked=anchors_checked,
        ignored_pointers=ignored_pointers,
        line_counts=line_counts,
        line_limits=limits,
        skipped_documents=tuple(sorted(set(skipped))),
        issues=tuple(issues),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--document", action="append", default=[], help="explicit Markdown file; repeatable")
    parser.add_argument("--include-frozen", action="store_true", help="include frozen/history trees")
    parser.add_argument(
        "--check-entrypoint-budgets",
        action="store_true",
        help="apply the stable root/entrypoint line budgets",
    )
    parser.add_argument("--line-limit", action="append", default=[], metavar="PATH=N", help="enforce a per-document line limit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = audit_repository(
            args.repo_root,
            documents=args.document or None,
            include_frozen=args.include_frozen,
            line_limits={
                **(ENTRYPOINT_LINE_LIMITS if args.check_entrypoint_budgets else {}),
                **_parse_limits(args.line_limit),
            },
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.ok else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
