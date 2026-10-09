"""Saved-artifact acceptance only; terminal planning and finalization are separate."""
import argparse
import json
from pathlib import Path
from molgap.gptrans_triplet_portability import accept
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    result = accept(args.output, args.inputs)
    atomic_json(args.report, result)
    if args.finalize:
        from molgap.constants import REPO_ROOT
        from molgap.gptrans_triplet_portability_records import close
        result["rml_finalization"] = close(REPO_ROOT, args.output.resolve(), args.inputs.resolve())
    print(json.dumps(result, indent=2))
