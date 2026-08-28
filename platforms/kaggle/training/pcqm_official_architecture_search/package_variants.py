"""Package bounded official-train-only PCQM architecture screens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = {
    "pairbias192x8": {
        "model_family": "pair_gps",
        "precision": "fp32",
        "hidden_channels": 192,
        "num_layers": 8,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "sum",
        "categorical_field_channels": 16,
        "pair_channels": 64,
        "path_steps": 5,
        "triplet_rank": 16,
        "atom_input_channels": 64,
        "bond_input_channels": 32,
        "batch_size": 64,
        "eval_batch_size": 64,
        "title": "PBFP32",
        "slug_variant": "pbfp32",
    },
    "deep160x11": {
        "hidden_channels": 160,
        "num_layers": 11,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "sum",
        "categorical_field_channels": 16,
        "title": "Deep160x11",
    },
    "edge96": {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 96,
        "categorical_encoder": "sum",
        "categorical_field_channels": 16,
        "title": "Edge96",
    },
    "fieldconcat16": {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "concat_project",
        "categorical_field_channels": 16,
        "title": "FieldConcat16",
    },
    "fieldconcat32": {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "concat_project",
        "categorical_field_channels": 32,
        "title": "FieldConcat32",
    },
    "radicalctx16": {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "sum",
        "categorical_field_channels": 16,
        "graph_context": "radical",
        "radical_context_channels": 16,
        "title": "RadicalContext16",
        "slug_variant": "radicalcontext16",
    },
    "radicalctx32": {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 64,
        "categorical_encoder": "sum",
        "categorical_field_channels": 16,
        "graph_context": "radical",
        "radical_context_channels": 32,
        "title": "RadicalContext32",
        "slug_variant": "radicalcontext32",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--owner", default="nothingnessvoid")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--mode", choices=("preflight", "train"), default="train")
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS)
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("run_variant.py").read_text(encoding="utf-8")
    for variant in args.variants:
        for seed in args.seeds:
            spec = VARIANTS[variant]
            run_config = {
                "format": "molgap-pcqm-official-architecture-config-v1",
                "variant": variant,
                "seed": seed,
                "mode": args.mode,
                "model_family": spec.get("model_family", "edge_state"),
                "precision": spec.get("precision", "amp"),
                "hidden_channels": spec["hidden_channels"],
                "num_layers": spec["num_layers"],
                "num_heads": spec["num_heads"],
                "edge_state_channels": spec["edge_state_channels"],
                "categorical_encoder": spec["categorical_encoder"],
                "categorical_field_channels": spec[
                    "categorical_field_channels"
                ],
                "graph_context": spec.get("graph_context", "none"),
                "radical_context_channels": spec.get(
                    "radical_context_channels", 16
                ),
                "pair_channels": spec.get("pair_channels", 64),
                "path_steps": spec.get("path_steps", 5),
                "triplet_rank": spec.get("triplet_rank", 16),
                "atom_input_channels": spec.get("atom_input_channels", 64),
                "bond_input_channels": spec.get("bond_input_channels", 32),
                "batch_size": spec.get("batch_size", 256),
                "eval_batch_size": spec.get("eval_batch_size", 512),
            }
            package = output_root / f"{variant}_seed{seed}_{args.mode}"
            package.mkdir(parents=True, exist_ok=True)
            script = template.replace(
                "RUN_CONFIG = None", f"RUN_CONFIG = {run_config!r}", 1
            )
            if script == template:
                raise RuntimeError("RUN_CONFIG placeholder was not replaced")
            (package / "run_variant.py").write_text(script, encoding="utf-8")
            slug_variant = spec.get("slug_variant", variant)
            mode_suffix = "-preflight" if args.mode == "preflight" else "-train"
            slug = (
                f"molgap-pcqm-arch-{slug_variant}-s{seed}{mode_suffix}-20260828"
            )
            metadata = {
                "id": f"{args.owner}/{slug}",
                "title": (
                    f"MolGap PCQM Arch {spec['title']} S{seed} "
                    f"{args.mode.title()} 20260828"
                ),
                "code_file": "run_variant.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "dataset_sources": [
                    "nothingnessvoid/molgap-pcqm-feature-screen-20260826",
                    "nothingnessvoid/molgap-pcqm-architecture-runtime-v8-20260828",
                ],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (package / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(package)


if __name__ == "__main__":
    main()
