"""Thin adapter to the shared execution-only diagnostic."""
from pathlib import Path
from molgap.family_execution_diagnostic import run

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run(here.parents[1], here / "config.json", here / "results")
