"""Static reuse navigation checks; no model imports or evidence-role access."""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = "docs/operations/EXPERIMENT_ADDON_GUIDE.md"
CLI = "docs/operations/EXPERIMENT_CLI.md"
WORKFLOW = "docs/operations/EXPERIMENT_WORKFLOW.md"
MODULAR_RUNTIME = "docs/operations/MODULAR_EXPERIMENT_RUNTIME.md"
BLOCK_READMES = tuple(
    f"production/{stage}/README.md" for stage in (
        "01_acquire", "02_graphs", "03_train", "04_evaluate",
        "05_delta_gw", "06_uq", "07_database", "history",
    )
) + ("platforms/_records/README.md", "platforms/_staging/README.md")
DOCUMENTS = (
    "README.md", "ARCHITECTURE.md", "AGENTS.md", "BRANCHES.md",
    "CURRENT_STATE.md", "ROADMAP.md", CLI, GUIDE, WORKFLOW, MODULAR_RUNTIME,
    "docs/operations/KAGGLE_WORKFLOW_SUBMISSION.md",
    "docs/operations/MODULAR_RUNTIME_DESKTOP_ACCEPTANCE.md",
    "docs/operations/SHARED_TRAINING_CORE_DESKTOP_ACCEPTANCE.md",
    "docs/operations/SHARED_TRAINING_CORE_SERVER_RETURN.md",
    "docs/operations/FAMILY_TRAINING_EXAMPLES.md",
    "docs/operations/FAMILY_TRAINING_CORE.md",
    "docs/operations/INFRASTRUCTURE_REPAIR_20261003.md",
    "docs/operations/RESEARCH_PROTOCOL.md", "docs/operations/BRANCH_HISTORY.md",
    "docs/operations/LOCAL_AUDIT_20261001.md",
    "production/README.md", "platforms/README.md", "models/REFERENCE_INDEX.md",
) + BLOCK_READMES
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _anchors(text: str) -> set[str]:
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    return {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in headings
    }


@pytest.mark.parametrize("relative", DOCUMENTS)
def test_local_navigation_links_resolve(relative):
    for link in LINK.findall(_read(relative)):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
            continue
        path, _, anchor = unquote(link).partition("#")
        target = ((ROOT / relative).parent / path).resolve() if path else ROOT / relative
        assert target.is_relative_to(ROOT), (relative, link)
        # A block may intentionally point to its thin CLI directory.
        assert target.is_file() or target.is_dir(), (relative, link)
        if anchor:
            assert target.is_file(), (relative, link)
            assert anchor in _anchors(target.read_text(encoding="utf-8")), (relative, link)


@pytest.mark.parametrize("module, callable_name", (
    ("src/molgap/pcqm_gptrans_v4.py", "run_training"),
    ("src/molgap/pcqm_official_edge_state.py", "train_official_edge_state"),
    ("src/molgap/edge_state_training_core.py", "save_checkpoint"),
    ("src/molgap/edge_state_training_core.py", "restore_checkpoint"),
    ("experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py", "paired_metrics"),
))
def test_named_reuse_callables_exist_without_importing_them(module, callable_name):
    text = _read(GUIDE)
    assert f"../../{module}" in LINK.findall(text)
    assert f"`{callable_name}`" in text or f"`save_checkpoint` / `{callable_name}`" in text
    tree = ast.parse(_read(module))
    assert callable_name in {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_block_navigation_is_not_hidden_by_gitignore():
    result = subprocess.run(
        ["git", "check-ignore", "--", *BLOCK_READMES], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
