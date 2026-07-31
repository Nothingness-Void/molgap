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
MODEL_RESULT_NAME = re.compile(
    r"^(?:routed_)?[a-z0-9]+(?:_[a-z0-9]+)*_(?:\d+k|\d+m)_v\d+$"
)
BANNED_ACTIVE_DIRECTORY = re.compile(
    r"(^|_)(new|latest|final|best|tmp|temp|misc|phase\d+)(_|$)"
)
CALENDAR_DATE = re.compile(r"(?:19|20)\d{6}")


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


def test_active_experiments_have_readme_entrypoints() -> None:
    missing = [
        path.name
        for path in (REPO_ROOT / "experiments").iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and not (path / "README.md").is_file()
    ]
    assert not missing, "active experiments missing README.md:\n" + "\n".join(missing)


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
