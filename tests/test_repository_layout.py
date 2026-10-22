from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from conftest import CLI_MODULES
from molgap.constants import REPO_ROOT

ACTIVE_ROOTS = ("production", "experiments", "platforms")
EXCLUDED_PARTS = {
    "history",       # frozen phase 1-7 reproduction
    "_closed",       # settled experiments
    "_retired",      # replaced production evidence
    "_records",      # downloaded remote payloads
    "_staging",      # generated source snapshots, not active implementation
    "packages",      # generated Kaggle copies, not source entrypoints
    "bundle",        # frozen remote wheel/notebook payloads
    "__pycache__",
}
ROOT_POINTER = re.compile(
    r"`((?:(?:production|experiments|platforms|models|src|tests|data)/[^`\n]+)"
    r"|(?:AGENTS|CURRENT_STATE|ROADMAP|TRACKS|ARCHITECTURE|NAMING|README)\.md)`"
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")
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
)

SLOT_DIR = "experiments/pcqm_k1_slot_readout_diagnostic"


def experiment_is_indexed(index: str, experiment: Path) -> bool:
    if f"`{experiment.name}/`" in index:
        return True
    owner = REPO_ROOT / "experiments/README.md"
    return any(
        target is not None and target.resolve().is_relative_to(experiment.resolve())
        for target in (resolve_doc_pointer(owner, p) for p in MARKDOWN_LINK.findall(index))
    )


def has_main_guard(source: str) -> bool:
    for node in ast.parse(source).body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        if isinstance(right, ast.Name):
            left, right = right, left
        if (isinstance(left, ast.Name) and left.id == "__name__"
                and isinstance(right, ast.Constant) and right.value == "__main__"):
            return True
    return False


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
    # Experiment navigation uses the index and owner README. Nested records and
    # dated STATUS histories may refer to retired source on their archive branch.
    for root in ACTIVE_ROOTS:
        directory = REPO_ROOT / root
        candidates = ([directory / "README.md", *directory.glob("*/README.md")]
                      if root == "experiments" else directory.rglob("README.md"))
        for path in candidates:
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
        # Import caches can remain after a tracked experiment is moved.
        and any(
            item.is_file()
            and not EXCLUDED_PARTS.intersection(item.relative_to(path).parts)
            for item in path.rglob("*")
        )
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


def test_active_experiments_have_indexed_entrypoints() -> None:
    missing = [
        path.name
        for path in active_experiment_dirs()
        if not any((path / name).is_file() for name in ("README.md", "protocol.md"))
    ]
    assert not missing, "active experiments missing an entrypoint:\n" + "\n".join(missing)
    index = (REPO_ROOT / "experiments/README.md").read_text(encoding="utf-8")
    unindexed = [path.name for path in active_experiment_dirs() if not experiment_is_indexed(index, path)]
    assert not unindexed, "experiments missing from index:\n" + "\n".join(unindexed)


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


def test_cli_aliases_are_complete_and_importable() -> None:
    # Aliases are loaded lazily. Detect a moved entry before its caller imports it.
    for alias, relative in CLI_MODULES.items():
        path = REPO_ROOT / relative
        assert path.is_file(), f"{alias} points to missing CLI: {relative}"
        spec = importlib.util.spec_from_file_location(f"probe_{alias}", path)
        assert spec is not None and spec.loader is not None


def test_argparse_clis_have_a_guarded_entrypoint() -> None:
    # Parsers may be constructed inline inside main(), so only require a guarded
    # entrypoint here. Representative subprocess tests below prove --help.
    failures: list[str] = []
    for path in argparse_clis():
        source = path.read_text(encoding="utf-8", errors="ignore")
        if not has_main_guard(source):
            failures.append(str(path.relative_to(REPO_ROOT)))
    assert not failures, "argparse CLIs missing main guard:\n" + "\n".join(failures)


@pytest.mark.parametrize("source, expected", (
    ("if __name__=='__main__': main()", True),
    ('if "__main__" == __name__: main()', True),
    ('# if __name__ == "__main__": main()', False),
    ('text = "__name__ == __main__"', False),
    ('if __name__ != "__main__": main()', False),
))
def test_main_guard_check_is_syntax_aware(source, expected):
    assert has_main_guard(source) is expected


def test_experiment_index_accepts_markdown_links_without_prefix_collisions():
    experiment = REPO_ROOT / SLOT_DIR
    assert experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic/README.md)", experiment)
    assert not experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic_other/README.md)", experiment)
    assert not experiment_is_indexed("[entry](pcqm_k1_slot_readout_diagnostic/../other/README.md)", experiment)


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
