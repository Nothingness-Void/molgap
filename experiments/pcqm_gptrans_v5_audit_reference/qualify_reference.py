"""Metadata-only additive qualification; no model or platform execution."""
import json
from pathlib import Path
from molgap.constants import REPO_ROOT

from molgap.gptrans_reference_qualification import qualify_reference


if __name__ == "__main__":
    print(json.dumps(qualify_reference(REPO_ROOT), indent=2))
