"""Accept retained audit output without constructing a model."""
import argparse
import json
from pathlib import Path
from molgap.gptrans_portability import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output, args.inputs), indent=2))
