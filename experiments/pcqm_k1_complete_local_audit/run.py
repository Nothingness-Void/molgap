"""Invoke shared component observations within the owning prospective scope."""
from pathlib import Path
from molgap.k1_component_diagnostic import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run(root, root / "results")
