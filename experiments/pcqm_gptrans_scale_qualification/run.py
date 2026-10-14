"""Invoke the shared qualification owner in a single-GPU fresh environment."""
import argparse
import json
from pathlib import Path
import sys

# The pinned fresh environment has no editable MolGap installation.
source_root = next((parent for parent in Path(__file__).resolve().parents
                    if (parent / "src/molgap/constants.py").is_file()), None)
if source_root is None:
    raise RuntimeError("Profile source package lacks the MolGap constants marker")
sys.path.insert(0, str(source_root / "src"))
from molgap.constants import REPO_ROOT
if REPO_ROOT != source_root:
    raise RuntimeError("Profile entrypoint resolved another MolGap source package")
from molgap.experiment_package import verify_experiment_source_package
from molgap.gptrans_scale_profile import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    print(json.dumps(profile(args.inputs, args.output, contract,
        source_identity=verify_experiment_source_package(args.source_package))))
