"""Local declarations only; shared prepare-release owns planning and packaging."""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_author_release import freeze_screen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_screen(Path(REPO_ROOT), args.cpu_output)))
