"""Saved-artifact diagnostic acceptance; never load or execute a model."""
import argparse
from pathlib import Path
from molgap.gptrans_bottleneck_records import accept_audit, close_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args()
    print(close_audit(Path.cwd(), args.output.resolve(), args.inputs.resolve()) if args.close
          else accept_audit(args.output.resolve(), args.inputs.resolve()))
