"""No-inference acceptance for the K3a descriptor-only causal control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accept_kunshan_vector_screen import accept_screen


CANDIDATE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_descriptor"
)


def accept(root: Path, source: str, cache_sha: str) -> dict:
    return accept_screen(
        root,
        source,
        completion_format="molgap-kunshan-conjugated-descriptor-screen-v1",
        candidate=CANDIDATE,
        candidate_parameter_count=3_672_257,
        baseline_delta={"conjugated_input": "none"},
        candidate_delta={
            "conjugated_input": "repeated_component_descriptor8",
            "descriptor_channels": 32,
            "injection_block": 2,
            "component_communication": "none",
            "return": "linear32x192_bias_free_zero_init",
        },
        report_format="molgap-kunshan-conjugated-descriptor-acceptance-v1",
        expected_cache=cache_sha,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cache-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = accept(args.root, args.source_commit, args.cache_sha)
    except Exception as error:
        result = {
            "accepted": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "model_inference_executed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
