"""Saved-record adapter only; never constructs a model or launches compute."""
import argparse
import json
from molgap.k1_relation_terminal import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--finalized-at", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.mode, args.finalized_at), indent=2))
