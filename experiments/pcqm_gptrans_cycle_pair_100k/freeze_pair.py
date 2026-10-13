"""Thin local scientific-input freezer; no training or submission."""
import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_cycle_release import freeze_pair

if __name__ == "__main__":
    print(json.dumps(freeze_pair(REPO_ROOT), indent=2))
