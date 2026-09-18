"""Thin CLI for matched-500K structural residual attribution."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_500k_module_attribution import analyze, atomic_json


EXPECTED_GRAPH_SHA = "3879dc27106034c27c893bac7d182d54fbbe93906efb83108e9dac541e0894ae"
EXPECTED_PREDICTION_SHA = {
    "edge_state": "c31fafb651fcfef08381627ac86f6639cae91f664ae7a39574d4ca0383b91ce0",
    "gptrans_t": "0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--edge-state", type=Path, required=True)
    parser.add_argument("--gptrans-t", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.graph,
        {
            "edge_state": args.edge_state,
            "gptrans_t": args.gptrans_t,
        },
        expected_hashes=EXPECTED_PREDICTION_SHA,
        expected_graph_hash=EXPECTED_GRAPH_SHA,
    )
    atomic_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
