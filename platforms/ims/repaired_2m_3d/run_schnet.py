"""Thin IMS adapter for repaired-2M SchNet preflight and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.repaired_2m_schnet import (
    Repaired2MSchNetConfig,
    accept_repaired_2m_schnet_embeddings,
    export_repaired_2m_schnet_embeddings,
    preflight_repaired_2m_schnet,
    stable_recovery_config,
    train_repaired_2m_schnet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=("primary", "augmented"), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--accept-embeddings", action="store_true")
    parser.add_argument("--stable-recovery", action="store_true")
    parser.add_argument("--recovery-checkpoint", type=Path)
    args = parser.parse_args()
    primary = args.root / "primary"
    secondary = args.root / "secondary"
    common = {
        "primary_graph_dir": primary / "graph_shards",
        "primary_acceptance": primary / "primary_acceptance.json",
        "secondary_graph_dir": (
            secondary / "graph_shards" if args.variant == "augmented" else None
        ),
        "secondary_acceptance": (
            secondary / "secondary_acceptance.json"
            if args.variant == "augmented"
            else None
        ),
    }
    if args.accept_embeddings:
        result = accept_repaired_2m_schnet_embeddings(
            root=args.root,
            output_path=args.root / "embeddings" / "acceptance.json",
        )
    elif args.export_embeddings:
        result = export_repaired_2m_schnet_embeddings(
            root=args.root,
            variant=args.variant,
        )
    elif args.preflight:
        config = (
            stable_recovery_config(args.variant)
            if args.stable_recovery
            else Repaired2MSchNetConfig(variant=args.variant)
        )
        result = preflight_repaired_2m_schnet(
            **common,
            variant=args.variant,
            output_path=args.root / "preflight" / f"{args.variant}.json",
            config=config,
        )
    else:
        config = (
            stable_recovery_config(args.variant)
            if args.stable_recovery
            else Repaired2MSchNetConfig(variant=args.variant)
        )
        result = train_repaired_2m_schnet(
            **common,
            output_dir=args.root / "outputs" / args.variant,
            config=config,
            recovery_checkpoint=args.recovery_checkpoint,
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
