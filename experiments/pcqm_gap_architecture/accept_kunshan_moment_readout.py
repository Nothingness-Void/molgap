"""No-inference acceptance for the K2 projected-moment readout screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accept_kunshan_vector_screen import accept_screen


CANDIDATE = (
    "ogb_distance_angle_triangle_edge_state_graph_state9_moment_readout32"
)


def accept(root: Path, source: str) -> dict:
    return accept_screen(
        root,
        source,
        completion_format="molgap-kunshan-moment-readout-screen-v1",
        candidate=CANDIDATE,
        candidate_parameter_count=3_684_753,
        baseline_delta={"moment_readout": "mean_only"},
        candidate_delta={
            "moment_readout": (
                "mean_plus_nonlinear_projected_first_centered_second"
            ),
            "moment_channels": 32,
            "return": "linear64x192_bias_free_zero_init",
        },
        report_format="molgap-kunshan-moment-readout-acceptance-v1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = accept(args.root, args.source_commit)
    except Exception as error:
        result = {
            "accepted": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "model_inference_executed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
