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
BLOCK_READMES = tuple(
    f"production/{stage}/README.md" for stage in (
        "01_acquire", "02_graphs", "03_train", "04_evaluate",
        "05_delta_gw", "06_uq", "07_database", "history",
    )
) + ("platforms/_records/README.md", "platforms/_staging/README.md")
DOCUMENTS = (
    "README.md", "ARCHITECTURE.md", "AGENTS.md", "BRANCHES.md",
    "CURRENT_STATE.md", "ROADMAP.md", CLI, GUIDE,
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


@pytest.mark.parametrize("relative", ("README.md", "ARCHITECTURE.md", CLI))
def test_entrypoints_link_directly_to_operation_map(relative):
    links = LINK.findall(_read(relative))
    expected = (
        "EXPERIMENT_ADDON_GUIDE.md#pick-the-operation"
        if relative == CLI else f"{GUIDE}#pick-the-operation"
    )
    assert expected in links
    if relative == "README.md":
        assert _read(relative).index(expected) < _read(relative).index("## Install")


@pytest.mark.parametrize("operation", (
    "Train GPTrans", "Train EdgeState", "Resume or infer from a checkpoint",
    "Analyze saved predictions", "Bind source and local observations",
    "Accept artifacts and qualify a comparison", "Close and index evidence",
))
def test_reuse_map_covers_operations(operation):
    table = _read(GUIDE).split("## Pick the Operation\n", 1)[1].split("## Shared Core", 1)[0]
    assert f"| {operation} |" in table


@pytest.mark.parametrize("module, callable_name", (
    ("src/molgap/pcqm_gptrans_v4.py", "run_training"),
    ("src/molgap/pcqm_gptrans_full_runner.py", "train_full"),
    ("src/molgap/pcqm_official_edge_state.py", "train_official_edge_state"),
    ("src/molgap/edge_state_training_core.py", "save_checkpoint"),
    ("src/molgap/edge_state_training_core.py", "restore_checkpoint"),
    ("src/molgap/pcqm_gptrans_full_acceptance.py", "accept"),
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


def test_cli_examples_use_selected_checkout():
    text = _read(CLI)
    assert not re.search(r"D:[/\\]w[/\\]cli", text, re.IGNORECASE)
    assert "$repoRoot = (Get-Location).Path" in text
    assert "--repo-root $repoRoot" in text
    assert ".\\.venv\\Scripts\\python.exe" in text


def test_block_navigation_is_not_hidden_by_gitignore():
    result = subprocess.run(
        ["git", "check-ignore", "--", *BLOCK_READMES], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("relative, limit", (
    ("AGENTS.md", 230),
    ("BRANCHES.md", 100),
    ("CURRENT_STATE.md", 120),
    ("ROADMAP.md", 120),
))
def test_root_routing_documents_stay_compact(relative, limit):
    lines = len(_read(relative).splitlines())
    assert lines <= limit, f"{relative}: {lines} lines exceeds navigation budget {limit}"


def test_authority_and_capability_routes_remain_explicit():
    text = _read(GUIDE)
    links = set(LINK.findall(text))
    assert {
        "../../src/molgap/inference.py", "../../models/README.md",
        "../../src/molgap/comparison_readiness.py", "EXPERIMENT_FAMILY_WORKFLOW.md",
        "MOLGAP_COMMON_DIRECTION_V5_FINAL.md", "../../research_memory/README.md",
        "../../research_memory/LIFECYCLE.md", "../../platforms/README.md",
    } <= links
    for skill in ("kaggle-molgap-workloads", "ims-molgap-workloads", "scnet-bw-dcu-molgap"):
        assert f"`{skill}`" in text
    assert "do not run it as prediction-only analysis" in text
    assert "does not load checkpoints or run inference" in text
    assert "never hand-edit derived indexes" in text
