"""Build self-contained Kaggle upload directories for architecture screens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = {
    "gps9_control": {
        "kind": "gps",
        "slug": "molgap-pc100k-gps9-control-20260824",
        "title": "MolGap PC100K GPS9 Control 20260824",
    },
    "structural_gps9": {
        "kind": "structural_gps",
        "slug": "molgap-pc100k-structural-gps9-20260824",
        "title": "MolGap PC100K Structural GPS9 20260824",
    },
    "structural_gap_only": {
        "kind": "structural_gps",
        "target_mode": "gap",
        "slug": "molgap-pc100k-structural-gap-only-r1",
        "title": "MolGap PC100K Structural Gap Only R1",
    },
    "normalized_rwse_gap": {
        "kind": "normalized_structural_gps",
        "target_mode": "gap",
        "rwse_alpha_init": 0.25,
        "slug": "molgap-pc100k-normalized-rwse-gap-r1",
        "title": "MolGap PC100K Normalized RWSE Gap R1",
    },
    "gated_structural_seed42": {
        "kind": "gated_structural_gps",
        "target_mode": "all",
        "seeds": [42],
        "runtime_archive": "resource_bounded_architecture_gated_r1.tar.gz",
        "slug": "molgap-pc100k-gated-structural-seed42-r1",
        "title": "MolGap PC100K Gated Structural Seed42 R1",
    },
    "gated_structural_seed43": {
        "kind": "gated_structural_gps",
        "target_mode": "all",
        "seeds": [43],
        "runtime_archive": "resource_bounded_architecture_gated_r1.tar.gz",
        "slug": "molgap-pc100k-gated-structural-seed43-r1",
        "title": "MolGap PC100K Gated Structural Seed43 R1",
    },
    "gated_structural_seed44": {
        "kind": "gated_structural_gps",
        "target_mode": "all",
        "seeds": [44],
        "runtime_archive": "resource_bounded_architecture_gated_r1.tar.gz",
        "slug": "molgap-pc100k-gated-structural-seed44-r1",
        "title": "MolGap PC100K Gated Structural Seed44 R1",
    },
    "edge_state_structural_seed42": {
        "kind": "edge_state_structural_gps",
        "target_mode": "all",
        "seeds": [42],
        "edge_state_channels": 64,
        "runtime_archive": "resource_bounded_architecture_edge_state_r1.tar.gz",
        "slug": "molgap-pc100k-edge-state-structural-seed42-r1",
        "title": "MolGap PC100K Edge State Structural Seed42 R1",
    },
    "edge_state_structural_seed43": {
        "kind": "edge_state_structural_gps",
        "target_mode": "all",
        "seeds": [43],
        "edge_state_channels": 64,
        "runtime_archive": "resource_bounded_architecture_edge_state_r1.tar.gz",
        "slug": "molgap-pc100k-edge-state-structural-seed43-r1",
        "title": "MolGap PC100K Edge State Structural Seed43 R1",
    },
    "edge_state_structural_seed44": {
        "kind": "edge_state_structural_gps",
        "target_mode": "all",
        "seeds": [44],
        "edge_state_channels": 64,
        "runtime_archive": "resource_bounded_architecture_edge_state_r1.tar.gz",
        "slug": "molgap-pc100k-edge-state-structural-seed44-r1",
        "title": "MolGap PC100K Edge State Structural Seed44 R1",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runtime-dataset", required=True)
    parser.add_argument("--owner", default="nothingnessvoid")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("run_screen.py").read_text(encoding="utf-8")

    for name in args.variants:
        spec = VARIANTS[name]
        package = output_root / name
        package.mkdir(parents=True, exist_ok=True)
        run_config = {
            "format": "molgap-resource-bounded-screen-config-v1",
            "name": name,
            "kind": spec["kind"],
            "target_mode": spec.get("target_mode", "all"),
            "rwse_alpha_init": spec.get("rwse_alpha_init", 0.25),
            "seeds": spec.get("seeds", [42, 43, 44]),
            "runtime_archive": spec.get(
                "runtime_archive",
                "resource_bounded_architecture_gap_rwse_r1.tar.gz",
            ),
            "edge_state_channels": spec.get("edge_state_channels", 64),
        }
        packaged_script = template.replace(
            "RUN_CONFIG = None",
            f"RUN_CONFIG = {run_config!r}",
            1,
        )
        if packaged_script == template:
            raise RuntimeError("RUN_CONFIG placeholder was not replaced")
        (package / "run_screen.py").write_text(packaged_script, encoding="utf-8")
        (package / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"{args.owner}/{spec['slug']}",
                    "title": spec["title"],
                    "code_file": "run_screen.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_internet": "true",
                    "dataset_sources": [args.dataset, args.runtime_dataset],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(package)


if __name__ == "__main__":
    main()
