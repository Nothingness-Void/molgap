"""Invoke the reusable frozen CPU diagnostic after prospective publication."""
from pathlib import Path
from molgap.k1_weight_average_diagnostic import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run(root, root / "results")
