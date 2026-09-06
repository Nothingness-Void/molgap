"""No-inference acceptance for the K3b ComponentState comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accept_kunshan_vector_screen import accept_screen


BASELINE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_descriptor"
)
CANDIDATE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_conjugated_component"
)
BASELINE_DELTA = {
    "conjugated_input": "repeated_component_descriptor8",
    "descriptor_channels": 32,
    "injection_block": 2,
    "component_communication": "none",
    "return": "linear32x192_bias_free_zero_init",
}
CANDIDATE_DELTA = {
    **BASELINE_DELTA,
    "component_communication": "persistent_component_state32",
    "component_update_blocks": [3, 6, 9],
    "atom_to_component": "mean_linear192x32_bias_free",
    "component_update": "shared_gated_residual32",
    "component_to_atom": "low_rank16_gated_linear192_zero_init",
}


def accept(root: Path, source: str, cache_sha: str) -> dict:
    return accept_screen(
        root,
        source,
        completion_format="molgap-kunshan-conjugated-component-screen-v1",
        candidate=CANDIDATE,
        candidate_parameter_count=3_694_033,
        baseline_delta=BASELINE_DELTA,
        candidate_delta=CANDIDATE_DELTA,
        report_format="molgap-kunshan-conjugated-component-acceptance-v1",
        expected_cache=cache_sha,
        baseline=BASELINE,
        baseline_parameter_count=3_672_257,
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
