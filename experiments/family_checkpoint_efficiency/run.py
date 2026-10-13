"""Thin local diagnostic adapter; no training dispatch or platform operations."""
import json
from pathlib import Path

from legacy_oracle import finite_state_dict, finite_tree
from molgap.checkpoint_diagnostic import run


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    result = run(here.parents[1], here / "config.json", here / "results",
        legacy_tree=finite_tree, legacy_state=finite_state_dict)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"],
        "cases": len(result["cases"]), "checks": result["checks"]}))
