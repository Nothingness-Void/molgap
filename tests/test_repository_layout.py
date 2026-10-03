from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from conftest import CLI_MODULES
from molgap.constants import REPO_ROOT
from molgap.documentation_check import ENTRYPOINT_LINE_LIMITS

ACTIVE_ROOTS = ("production", "experiments", "platforms")
EXCLUDED_PARTS = {
    "history",       # frozen phase 1-7 reproduction
    "_closed",       # settled experiments
    "_retired",      # replaced production evidence
    "_records",      # downloaded remote payloads
    "_staging",      # generated remote source bundles
    "packages",      # generated Kaggle copies, not source entrypoints
    "bundle",        # frozen remote wheel/notebook payloads
    "__pycache__",
}
STALE_PATH_SNIPPETS = (
    '"results/phase8',
    '"scripts/phase8',
    "'results/phase8",
    "'scripts/phase8",
)
MODEL_RESULT_NAME = re.compile(
    r"^(?:routed_)?[a-z0-9]+(?:_[a-z0-9]+)*_(?:\d+k|\d+m)_v\d+$"
)
BANNED_ACTIVE_DIRECTORY = re.compile(
    r"(^|_)(new|latest|final|best|tmp|temp|misc|phase\d+)(_|$)"
)
CALENDAR_DATE = re.compile(r"(?:19|20)\d{6}")
ROOT_POINTER = re.compile(
    r"`((?:(?:production|experiments|platforms|models|src|tests|data)/[^`\n]+)"
    r"|(?:AGENTS|CURRENT_STATE|ROADMAP|TRACKS|ARCHITECTURE|NAMING|README)\.md)`"
)
LOCAL_ARTIFACT_POINTER = re.compile(r"`([^`\n]+\.(?:md|json))`")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")
TRACEABILITY_ROLE = re.compile(
    r"^\s*Traceability role:\s*`(evidence|context|diagnostic|pending|portfolio)`",
    re.IGNORECASE | re.MULTILINE,
)
NON_EVIDENCE_ROLES = {"context", "diagnostic", "pending", "portfolio"}
REPO_POINTER_ROOTS = {
    "data",
    "experiments",
    "models",
    "platforms",
    "production",
    "src",
    "tests",
}
REPO_CONTROL_DOCS = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CURRENT_STATE.md",
    "NAMING.md",
    "README.md",
    "ROADMAP.md",
    "TRACKS.md",
}
LOCAL_LINK_SUFFIXES = (
    ".csv",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".py",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
)
CONTROL_DOC_LIMITS = ENTRYPOINT_LINE_LIMITS
TOP_LEVEL_POINTER_DOCS = (
    "AGENTS.md",
    "CURRENT_STATE.md",
    "ROADMAP.md",
    "TRACKS.md",
    "ARCHITECTURE.md",
    "NAMING.md",
    "README.md",
    "experiments/README.md",
    "production/README.md",
    "platforms/README.md",
    "experiments/DIRECTORY_INDEX.md",
    "experiments/EVIDENCE_INDEX.md",
    "docs/operations/EXPERIMENT_QUICKSTART.md",
    "docs/operations/GRAPH_SCREEN_EXTENSION.md",
    "docs/operations/ARCHITECTURE_MODULE_INDEX.md",
    "docs/operations/CURRENT_STATE_HISTORY_INDEX.md",
    "docs/operations/ROADMAP_HISTORY_INDEX.md",
)
EXPERIMENT_DIRECTORY_INDEX = REPO_ROOT / "experiments" / "DIRECTORY_INDEX.md"
EXPERIMENT_EVIDENCE_INDEX = REPO_ROOT / "experiments" / "EVIDENCE_INDEX.md"
INDEXED_EXPERIMENT = re.compile(r"`([a-z0-9][a-z0-9_\-]*)/`")


def active_python_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def active_markdown_files() -> list[Path]:
    files = [
        REPO_ROOT / name
        for name in (
            "AGENTS.md",
            "ARCHITECTURE.md",
            "CURRENT_STATE.md",
            "NAMING.md",
            "README.md",
            "ROADMAP.md",
            "TRACKS.md",
        )
    ]
    for root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.md"):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(set(files))


def argparse_clis() -> list[Path]:
    return [
        path
        for path in active_python_files()
        if "argparse" in path.read_text(encoding="utf-8", errors="ignore")
    ]


def active_experiment_dirs() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )


def resolve_doc_pointer(document: Path, raw_pointer: str) -> Path | None:
    pointer = raw_pointer.split("#", maxsplit=1)[0]
    if any(token in pointer for token in ("*", "<", ">")):
        return None
    normalized = pointer.replace("\\", "/")
    first = normalized.split("/", maxsplit=1)[0]
    if first in REPO_POINTER_ROOTS or normalized in REPO_CONTROL_DOCS:
        return REPO_ROOT / normalized
    return document.parent / normalized


def markdown_targets(document: Path) -> list[str]:
    """Return local Markdown targets without treating prose code as links."""

    targets: list[str] = []
    text = document.read_text(encoding="utf-8", errors="ignore")
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip()
        if target.startswith("<") and ">" in target:
            target = target[1 : target.index(">")]
        else:
            # Optional Markdown link titles are outside the path token.
            target = target.split(None, maxsplit=1)[0]
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append(target)
    return targets


def existing_local_pointers(document: Path) -> set[Path]:
    """Resolve explicit links and only reliable, existing inline path pointers."""

    pointers = markdown_targets(document)
    text = document.read_text(encoding="utf-8", errors="ignore")
    pointers.extend(LOCAL_ARTIFACT_POINTER.findall(text))
    resolved: set[Path] = set()
    for raw_pointer in pointers:
        if raw_pointer.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = raw_pointer.strip().split(None, maxsplit=1)[0]
        path = resolve_doc_pointer(document, target)
        if path is not None and path.is_file():
            resolved.add(path.resolve())
    return resolved


def traceability_role(readme: Path) -> str:
    match = TRACEABILITY_ROLE.search(
        readme.read_text(encoding="utf-8", errors="ignore")
    )
    return match.group(1).lower() if match else "evidence"


def test_active_experiments_have_readme_entrypoints() -> None:
    """The conditional directory index is the stable entrypoint for every question."""

    directory_index = EXPERIMENT_DIRECTORY_INDEX.read_text(encoding="utf-8")
    indexed = set(INDEXED_EXPERIMENT.findall(directory_index))
    active_dirs = active_experiment_dirs()
    active = {path.name for path in active_dirs}
    missing_readmes = sorted(
        path.name for path in active_dirs if not (path / "README.md").is_file()
    )
    assert not missing_readmes, (
        "active experiments missing README.md entrypoints:\n"
        + "\n".join(missing_readmes)
    )
    assert indexed == active, (
        "experiments/DIRECTORY_INDEX.md must cover exactly active question directories:\n"
        f"missing={sorted(active - indexed)}\nextra={sorted(indexed - active)}"
    )


def test_active_experiments_are_indexed_and_traceable() -> None:
    """Validate conditional indexes and retain the accepted-evidence gate."""

    directory_index = EXPERIMENT_DIRECTORY_INDEX.read_text(encoding="utf-8")
    evidence_index = EXPERIMENT_EVIDENCE_INDEX.read_text(encoding="utf-8")
    indexed_directories = set(INDEXED_EXPERIMENT.findall(directory_index))
    indexed_evidence = set(INDEXED_EXPERIMENT.findall(evidence_index))
    active_names = {path.name for path in active_experiment_dirs()}
    failures: list[str] = []
    for name in sorted(indexed_directories - active_names):
        failures.append(f"{name}: DIRECTORY_INDEX.md points to missing directory")
    for name in sorted(indexed_evidence - active_names):
        failures.append(f"{name}: EVIDENCE_INDEX.md points to missing directory")

    # An explicit role marker is used when an indexed question is a
    # portfolio/context/diagnostic or pending record rather than a terminal
    # evidence bundle. The default role is ``evidence`` so accepted rows keep
    # the historical decision/evidence gate.
    for name in sorted(indexed_evidence & active_names):
        experiment = REPO_ROOT / "experiments" / name
        readme = experiment / "README.md"
        role = traceability_role(readme) if readme.is_file() else "evidence"
        if role not in NON_EVIDENCE_ROLES:
            decisions = sorted(experiment.rglob("*decision*.md"))
            evidence = sorted(experiment.rglob("*.json"))
            if not decisions:
                failures.append(f"{name}: no *decision*.md")
                continue
            decision_paths = {path.resolve() for path in decisions}
            if not (existing_local_pointers(readme) & decision_paths):
                failures.append(f"{name}: README does not point to a decision")
            if not evidence:
                failures.append(f"{name}: no JSON machine evidence")
            else:
                evidence_documents = [readme, *decisions]
                if not any(
                    path.suffix.lower() == ".json"
                    and path.resolve() in existing_local_pointers(document)
                    for document in evidence_documents
                    for path in evidence
                ):
                    failures.append(
                        f"{name}: README or decision has no reachable JSON evidence"
                    )

    assert not failures, "experiment traceability failures:\n" + "\n".join(failures)


def test_live_control_documents_remain_bounded() -> None:
    failures: list[str] = []
    for relative, limit in CONTROL_DOC_LIMITS.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        line_count = len(text.splitlines())
        if line_count > limit:
            failures.append(f"{relative}: {line_count} lines exceeds {limit}")
    assert not failures, "live control documents accumulated history:\n" + "\n".join(failures)


def test_top_level_document_pointers_exist() -> None:
    failures: list[str] = []
    for relative in TOP_LEVEL_POINTER_DOCS:
        document = REPO_ROOT / relative
        text = document.read_text(encoding="utf-8", errors="ignore")
        for match in ROOT_POINTER.finditer(text):
            pointer = match.group(1).split("#", maxsplit=1)[0].rstrip("/")
            if any(token in pointer for token in ("*", "<", ">")):
                continue
            if not (REPO_ROOT / pointer).exists():
                failures.append(f"{relative}: {match.group(1)}")
    assert not failures, "missing top-level document pointers:\n" + "\n".join(failures)


def test_active_experiment_entrypoint_pointers_exist() -> None:
    failures: list[str] = []
    for experiment in active_experiment_dirs():
        documents = [experiment / "README.md"]
        status = experiment / "STATUS.md"
        if status.is_file():
            documents.append(status)
        for document in documents:
            for raw_pointer in markdown_targets(document):
                resolved = resolve_doc_pointer(document, raw_pointer)
                if resolved is not None and not resolved.exists():
                    failures.append(
                        f"{document.relative_to(REPO_ROOT)}: {raw_pointer}"
                    )
    assert not failures, "missing experiment entrypoint pointers:\n" + "\n".join(failures)


def test_active_markdown_links_resolve() -> None:
    failures: list[str] = []
    for document in active_markdown_files():
        text = document.read_text(encoding="utf-8", errors="ignore")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", maxsplit=1)[0]
            if not target:
                continue
            normalized = target.replace("\\", "/")
            first = normalized.split("/", maxsplit=1)[0]
            looks_local = (
                target.startswith(("./", "../"))
                or first in REPO_POINTER_ROOTS
                or normalized in REPO_CONTROL_DOCS
                or normalized.lower().endswith(LOCAL_LINK_SUFFIXES)
            )
            if not looks_local:
                continue
            resolved = resolve_doc_pointer(document, target)
            if resolved is not None and not resolved.exists():
                failures.append(f"{document.relative_to(REPO_ROOT)}: {raw_target}")
    assert not failures, "broken active Markdown links:\n" + "\n".join(failures)


def test_production_model_result_names_are_self_describing() -> None:
    train_root = REPO_ROOT / "production" / "03_train"
    failures = [
        path.name
        for path in train_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and path.name != "scripts"
        and MODEL_RESULT_NAME.fullmatch(path.name) is None
    ]
    assert not failures, (
        "production model directories must encode composition, training rows, "
        "and version:\n" + "\n".join(failures)
    )


def test_active_directories_use_role_or_question_names() -> None:
    failures: list[str] = []
    for root_name in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root_name).rglob("*"):
            if not path.is_dir():
                continue
            relative = path.relative_to(REPO_ROOT)
            if EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if BANNED_ACTIVE_DIRECTORY.search(path.name) or CALENDAR_DATE.search(path.name):
                failures.append(str(relative))
    assert not failures, (
        "active directories contain ambiguous, phase, or calendar names:\n"
        + "\n".join(failures)
    )


def test_cli_aliases_are_complete_and_importable() -> None:
    # conftest imports every alias during collection; this additionally makes a
    # missing/moved file failure explicit instead of silently skipping it.
    for alias, relative in CLI_MODULES.items():
        path = REPO_ROOT / relative
        assert path.is_file(), f"{alias} points to missing CLI: {relative}"
        spec = importlib.util.spec_from_file_location(f"probe_{alias}", path)
        assert spec is not None and spec.loader is not None


def test_active_sources_do_not_encode_removed_phase_trees() -> None:
    failures: list[str] = []
    for path in active_python_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in STALE_PATH_SNIPPETS:
            if snippet in source:
                failures.append(f"{path.relative_to(REPO_ROOT)}: {snippet}")
    assert not failures, "removed tree references:\n" + "\n".join(failures)


def test_active_sources_do_not_infer_repo_root_from_parent_depth() -> None:
    failures: list[str] = []
    for path in active_python_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            text = ast.get_source_segment(source, node) or ""
            if "Path(__file__)" in text and ".parents[" in text:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', '?')}"
                )
    assert not failures, (
        "active code derives repository roots from directory depth; "
        "import REPO_ROOT from molgap.constants instead:\n" + "\n".join(failures)
    )


def test_argparse_clis_have_a_guarded_entrypoint() -> None:
    # Parsers may be constructed inline inside main(), so only require a guarded
    # entrypoint here. Representative subprocess tests below prove --help.
    failures: list[str] = []
    for path in argparse_clis():
        source = path.read_text(encoding="utf-8", errors="ignore")
        has_guard = '__name__ == "__main__"' in source or "__name__ == '__main__'" in source
        if not has_guard:
            failures.append(str(path.relative_to(REPO_ROOT)))
    assert not failures, "argparse CLIs missing main guard:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    "relative",
    [
        "production/03_train/scripts/training/train_encoder.py",
        "production/04_evaluate/scripts/evaluation/build_model_report_table.py",
        "production/04_evaluate/scripts/evaluation/benchmark_routed_v4_latency.py",
        "production/04_evaluate/scripts/evaluation/benchmark_repaired_2m_latency.py",
        "production/04_evaluate/scripts/evaluation/smoke_test_repaired_2m_inference.py",
        "production/04_evaluate/scripts/evaluation/verify_repaired_2m_public_inference.py",
        "production/04_evaluate/scripts/evaluation/compare_dft_vs_ml_cost.py",
        "production/04_evaluate/scripts/evaluation/benchmark_experimental_values.py",
        "production/04_evaluate/scripts/evaluation/build_presentation_evidence.py",
        "production/07_database/scripts/build_database.py",
        "experiments/pcqm_route_b/build_pcqm_route_b_1m.py",
        "experiments/pcqm_gine_expert/continue_pcqm_gine_local.py",
        "experiments/_scripts/run_conformer_ab.py",
        "platforms/kaggle/organize_account.py",
    ],
)
def test_representative_cli_help(relative: str) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{relative}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "usage:" in result.stdout.lower()
