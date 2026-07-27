from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from conftest import CLI_MODULES
from molgap.constants import REPO_ROOT

ACTIVE_ROOTS = ("production", "experiments", "platforms")
EXCLUDED_PARTS = {
    "history",       # frozen phase 1-7 reproduction
    "_closed",       # settled experiments
    "_records",      # downloaded remote payloads
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


def active_python_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def argparse_clis() -> list[Path]:
    return [
        path
        for path in active_python_files()
        if "argparse" in path.read_text(encoding="utf-8", errors="ignore")
    ]


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
        "production/04_evaluate/scripts/evaluation/build_phase8_report_table.py",
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
