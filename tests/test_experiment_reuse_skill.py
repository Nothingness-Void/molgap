"""Repository-skill integrity checks; no model imports or remote operations."""
import ast
from pathlib import Path
import re

import pytest
import yaml

from molgap.constants import REPO_ROOT


SKILL = REPO_ROOT / ".agents/skills/molgap-experiment-reuse"


def test_skill_discovery_metadata():
    raw = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    header = yaml.safe_load(raw.split("---", 2)[1])
    assert header["name"] == SKILL.name
    assert header["description"] and len(header["description"]) <= 1024
    metadata = yaml.safe_load((SKILL / "agents/openai.yaml").read_text(encoding="utf-8"))
    assert metadata["policy"]["allow_implicit_invocation"] is True
    assert "$"+header["name"] in metadata["interface"]["default_prompt"]


@pytest.mark.parametrize("relative", ["SKILL.md", "references/entrypoints.md"])
def test_routing_links_resolve_within_checkout(relative):
    path = SKILL / relative
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
    assert links
    for link in links:
        target = (path.parent / link).resolve()
        assert target.is_relative_to(REPO_ROOT.resolve()), link
        assert target.is_file(), link


def test_indexed_public_entrypoints_exist_without_importing_runtimes():
    text = (SKILL / "references/entrypoints.md").read_text(encoding="utf-8")
    rows = [row.split("|")[2] for row in text.splitlines()
            if row.startswith("| ") and ".py)" in row]
    assert rows
    for row in rows:
        segments = re.split(r"(?=\[[^\]]+\]\([^)]*\.py\):)", row)
        for segment in segments:
            match = re.match(r"\[[^\]]+\]\(([^)]+)\):(.+)", segment)
            if not match:
                continue
            path = (SKILL / "references" / match.group(1)).resolve()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            declared = {node.name for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            for name in re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", match.group(2)):
                assert name in declared, f"{path.name}:{name} no longer exists"
