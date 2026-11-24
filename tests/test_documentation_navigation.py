"""Static reuse navigation checks; no model imports or evidence-role access."""
from __future__ import annotations

import argparse
import ast
import json
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
EXTRA_PAGES = (
    "experiments/README.md", "production/README.md", "platforms/README.md", "models/REFERENCE_INDEX.md",
) + BLOCK_READMES
DOCUMENTS = tuple(sorted({
    p.relative_to(ROOT).as_posix() for pattern in ("*.md", "docs/**/*.md", "research_memory/*.md", "skills/**/*.md")
    for p in ROOT.glob(pattern)
} | set(EXTRA_PAGES)))
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


# --- Durable documentation: stale text must fail a test ---
CALL = re.compile(r"(?:(?P<rml>rml)|molgap\.research_memory|(?P<cli>cli|molgap\.experiment_cli|experiment_cli))"
                  r"(?:\s+--)?(?:\s+--repo-root\s+\S+)?\s+(?P<sub>[a-z][a-z-]+)(?P<rest>.*)$")
MACHINE_PATHS = ("D:/文档/molgap/", "D:/molgap-machine-profile.json", "D:/molgap-artifacts", "D:/molgap-worktrees/", "D:/下载/Key")
SINGLE_SOURCE = (  # fact -> the one page that may state it (listed in docs/README.md)
    (r"python(?:\.exe)? platforms/local/infra_budget\.py", "AGENTS.md"),
    (r"128,606", "AGENTS.md"),
    (r"D:/下载/Key", "skills/kaggle-molgap-workloads/references/account-and-cli.md"),
)


def _subcommands() -> dict[str, dict[str, set[str]]]:
    from molgap.experiment_cli import _parser
    from molgap.research_memory.cli import build_parser
    table = {}
    for tool, parser in (("rml", build_parser()), ("cli", _parser())):
        flags = lambda p: {o for a in p._actions for o in a.option_strings}
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        table[tool] = {name: flags(p) | flags(parser) for name, p in sub.choices.items()}
    return table


@pytest.mark.parametrize("relative", DOCUMENTS)
def test_documented_commands_exist(relative):
    known = _subcommands()
    for span in re.findall(r"`([^`\n]+)`", _read(relative)):
        found = CALL.search(span)
        if found:
            tool = "cli" if found["cli"] else "rml"
            assert found["sub"] in known[tool], (relative, span)
            for flag in re.findall(r"--[a-z][a-z-]*", found["rest"]):
                assert flag in known[tool][found["sub"]], (relative, span, flag)


def test_lifecycle_map_commands_exist():
    text = _read("docs/operations/LIFECYCLE_MAP.md")
    for row in re.findall(r"^\| (?:Research|Plan|Propose|Accept|Archive) \| ([^|]+)\|", text, re.MULTILINE):
        script = re.match(r"`(platforms/[^`\s]+\.py)`", row)
        if script:
            source = _read(script[1])
            assert all(flag in source for flag in re.findall(r"--[a-z][a-z-]*", row)), row
    named = set(re.findall(r"`(?:rml |cli )?([a-z][a-z-]+)", text))
    missing = [f"{tool} {name}" for tool, names in _subcommands().items() for name in names if name not in named]
    assert not missing, f"commands with no lifecycle map row: {missing}"
    # The "Other ..." lines may skip the table only if the command is documented elsewhere.
    reference = "".join(_read(p) for p in ("docs/operations/EXPERIMENT_CLI.md", "docs/operations/RML_INTERFACES.md", "docs/operations/RESEARCH_PROTOCOL.md", "AGENTS.md"))
    for line in re.findall(r"^Other `\w+` commands.*$", text, re.MULTILINE):
        assert all(name in reference for name in re.findall(r"`([a-z][a-z-]+)`", line.split(":", 1)[-1])), line


@pytest.mark.parametrize("relative", DOCUMENTS)
def test_pages_use_no_absolute_paths_or_copied_facts(relative):
    text = _read(relative)
    assert "file:///" not in text, relative
    assert all(p.startswith(MACHINE_PATHS) for p in re.findall(r"D:/[^\s`'\")]*", text)), relative
    for fact, owner in SINGLE_SOURCE:
        assert relative == owner or not re.search(fact, text), (relative, fact, owner)
    for path, function in re.findall(r"Verified by: (tests/\S+?\.py)::(\w+)", text):
        assert function in {n.name for n in ast.walk(ast.parse(_read(path))) if isinstance(n, ast.FunctionDef)}, relative


def test_docs_are_undated_and_current_state_ids_are_open():
    assert not [p for p in (ROOT / "docs").rglob("*") if re.search(r"\d{8}", p.name)]
    index = {t["trajectory_id"]: t["outcome"] for t in json.loads(
        _read("research_memory/derived/trajectory_index.json"))["trajectories"]}
    for row in _read("CURRENT_STATE.md").splitlines():
        for tid in re.findall(r"`(TB?-[^`\s]+)`", row):
            # An id missing here is allowed only when another branch owns the record.
            assert index.get(tid, "ACTIVE") == "ACTIVE", (tid, index.get(tid))
            assert tid in index or "`codex/exp/" in row, f"unknown trajectory id {tid}"
