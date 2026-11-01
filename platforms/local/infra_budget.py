"""Report infrastructure size against docs/operations/INFRA_BUDGET.json.

Exit 1 when any metric exceeds its limit. Standalone: no experiment flow calls it.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUDGET = ROOT / "docs/operations/INFRA_BUDGET.json"
VERSION = re.compile(r"molgap-[a-z0-9-]+-v[0-9]+")


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def metrics():
    src = sorted((ROOT / "src/molgap").rglob("*.py"))
    docs = sorted((ROOT / "docs/operations").glob("*.md"))
    tests = sorted((ROOT / "tests").glob("*.py"))
    texts = {path: read(path) for path in src}
    lines = lambda paths: sum(len(read(p).splitlines()) for p in paths)
    return {
        "src_py_files": len(src),
        "src_py_lines": sum(len(t.splitlines()) for t in texts.values()),
        "research_memory_lines": sum(len(t.splitlines()) for p, t in texts.items()
                                     if "research_memory" in p.relative_to(ROOT / "src/molgap").parts[:1]),
        "src_hashlib_files": sum("hashlib" in t for t in texts.values()),
        "gitattributes_text_rules": sum("-text" in l for l in read(ROOT / ".gitattributes").splitlines()),
        "test_files": len(tests),
        "test_files_mentioning_hash": sum(bool(re.search("sha256|hashlib", read(p))) for p in tests),
        "format_version_strings": len(set(VERSION.findall("\n".join(texts.values())))),
        "docs_operations_files": len(docs),
        "docs_operations_lines": lines(docs),
    }


def main():
    now = metrics()
    limits = json.loads(read(BUDGET))["limits"]
    over = {k: (v, limits[k]) for k, v in now.items() if k in limits and v > limits[k]}
    for key, value in now.items():
        print(f"{key:30} {value:>8} / {limits.get(key, '-')}")
    for key, (value, limit) in over.items():
        print(f"OVER BUDGET: {key} {value} > {limit}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
