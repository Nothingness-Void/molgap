"""Call the owning mechanical retention and prospective release adapter."""
import argparse
import json
from pathlib import Path
from molgap.constants import REPO_ROOT
from molgap.gptrans_triplet_portability_prepare import freeze_assets, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_assets(REPO_ROOT) if args.freeze_assets else prepare(REPO_ROOT, args.output.resolve()), indent=2))
