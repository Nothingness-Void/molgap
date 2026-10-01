"""Metadata-only additive qualification; no model or platform execution."""
import json
from pathlib import Path

from molgap.gptrans_reference_qualification import qualify_reference


if __name__ == "__main__":
    print(json.dumps(qualify_reference(Path(__file__).resolve().parents[2]), indent=2))
