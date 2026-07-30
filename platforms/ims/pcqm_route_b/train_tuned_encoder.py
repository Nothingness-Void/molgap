"""Train one full-scale Route B encoder with its confirmed search winner."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from molgap.pcqm_route_b_training import (
    CONFIGS,
    TrainingOverrides,
    train_encoder,
)


WARM_STARTS = {
    "gps9": "gps9.pt",
    "gps11_160": "gps11_160.pt",
    "primary_schnet": "primary_schnet.pt",
    "augmented_schnet": "augmented_schnet.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(CONFIGS), required=True)
    args = parser.parse_args()
    summary_path = (
        args.root
        / "search"
        / "outputs"
        / "100k_confirm"
        / args.encoder
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("official_valid_metric_read") is not False
        or summary.get("official_test_used") is not False
        or summary.get("sealed_20k_used") is not False
    ):
        raise RuntimeError(f"Search winner is not development-only: {summary_path}")
    trial = summary["ranking"][0]["trial"]
    base = CONFIGS[args.encoder]
    config = replace(
        base,
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
    graph_root = args.root / "inputs" / "pcqm_route_b_1m"
    roots = {config.modality: graph_root}
    if config.augmented:
        roots["secondary"] = graph_root
    result = train_encoder(
        config=config,
        roots=roots,
        warm_start=args.root / "warmstarts" / WARM_STARTS[args.encoder],
        output_dir=args.root / "outputs_tuned" / args.encoder,
        overrides=overrides,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
