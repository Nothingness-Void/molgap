"""SCNet adapter for one tuned full-scale PCQM Route B GPS encoder."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from molgap.pcqm_route_b_training import (
    CONFIGS,
    TrainingOverrides,
    preflight_encoder,
    train_encoder,
)


WARM_STARTS = {
    "gps9": "gps9.pt",
    "gps11_160": "gps11_160.pt",
}


def _configuration(root: Path, encoder: str):
    summary_path = root / "search_winners" / f"{encoder}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("official_valid_metric_read") is not False
        or summary.get("official_test_used") is not False
        or summary.get("sealed_20k_used") is not False
    ):
        raise RuntimeError(f"invalid development-only winner: {summary_path}")
    trial = summary["ranking"][0]["trial"]
    config = replace(
        CONFIGS[encoder],
        batch_size=int(trial["batch_size"]),
        learning_rate=float(trial["learning_rate"]),
        weight_decay=float(trial["weight_decay"]),
    )
    overrides = TrainingOverrides(
        dropout=float(trial["dropout"]),
        warmup_ratio=float(trial["warmup_ratio"]),
        grad_clip=float(trial["grad_clip"]),
        schnet_cutoff=6.0,
    )
    return config, overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(WARM_STARTS), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config, overrides = _configuration(args.root, args.encoder)
    graph_root = args.root / "inputs" / "pcqm_route_b_1m"
    if args.preflight:
        result = preflight_encoder(
            config=config,
            root=graph_root,
            warm_start=args.root / "warmstarts" / WARM_STARTS[args.encoder],
            output_path=args.root / "preflight" / args.encoder,
            overrides=overrides,
        )
    else:
        result = train_encoder(
            config=config,
            roots={"gps": graph_root},
            warm_start=args.root / "warmstarts" / WARM_STARTS[args.encoder],
            output_dir=args.root / "outputs_tuned" / args.encoder,
            overrides=overrides,
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
