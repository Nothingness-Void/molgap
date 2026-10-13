"""Thin local adapter; never submits or opens development roles."""
import argparse
from pathlib import Path

from molgap.input_pipeline_diagnostic import run
from molgap.training_reproducibility import atomic_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = Path(__file__).resolve().parent
    try:
        run(root, folder / "config.json", folder / "results", gpu=args.gpu)
    except Exception as error:
        atomic_json(folder / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
