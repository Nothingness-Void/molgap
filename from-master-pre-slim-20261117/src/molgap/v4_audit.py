"""Fail-closed static audit for active V4 training paths."""
from __future__ import annotations

import ast
import re
from pathlib import Path


RULES = {
    "std-correction": "Use v4_runtime.sample_std_compat",
    "direct-adamw": "Use v4_runtime.make_adamw_compat",
    "direct-torch-load": "Use v4_runtime.torch_load_compat",
    "accelerator-bincount": "Replace or explicitly certify accelerator torch.bincount",
}


def _call_name(node: ast.Call) -> str:
    parts = []
    current = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def audit_python_file(path: Path) -> list[dict]:
    if path.name == "v4_runtime.py":
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings = []
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"torch", "torch.optim"}:
            for item in node.names:
                if node.module == "torch" and item.name in {"load", "bincount"}:
                    aliases[item.asname or item.name] = f"torch.{item.name}"
                elif node.module == "torch.optim" and item.name == "AdamW":
                    aliases[item.asname or item.name] = "torch.optim.AdamW"

    def resolved_name(node: ast.Call) -> str:
        name = _call_name(node)
        root, separator, rest = name.partition(".")
        if root in aliases:
            return aliases[root] + (separator + rest if separator else "")
        return name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolved_name(node)
        keywords = {keyword.arg for keyword in node.keywords}
        rule = None
        if (name.endswith(".std") or name == "std") and "correction" in keywords:
            rule = "std-correction"
        elif name in {"torch.optim.AdamW", "optim.AdamW"}:
            rule = "direct-adamw"
        elif name == "torch.load":
            rule = "direct-torch-load"
        elif name == "torch.bincount":
            rule = "accelerator-bincount"
        if rule:
            findings.append(
                {
                    "rule": rule,
                    "path": path.as_posix(),
                    "line": int(node.lineno),
                    "message": RULES[rule],
                }
            )
    return findings


def audit_v4_paths(paths) -> dict:
    files = [Path(path) for path in paths]
    findings = []
    for path in files:
        if not path.is_file():
            findings.append(
                {"rule": "missing-file", "path": path.as_posix(), "line": 0, "message": "Declared V4 source is missing"}
            )
            continue
        findings.extend(audit_python_file(path))
    return {
        "format": "molgap-v4-static-audit-v1",
        "status": "accepted" if not findings else "rejected",
        "files": [path.as_posix() for path in files],
        "findings": findings,
    }


def require_v4_audit(paths) -> dict:
    result = audit_v4_paths(paths)
    if result["status"] != "accepted":
        raise RuntimeError(f"V4 static audit failed: {result['findings']}")
    return result
