"""Create three bounded Kaggle fusion kernel packages after staging publication."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


VARIANTS = ("coverage2m", "hard20k", "multi2d")
KERNEL_IDS = {
    "coverage2m": "nothingnessvoid/molgap-2m-2d-1m-3d-fusion-coverage2m",
    "hard20k": "nothingnessvoid/molgap-2m-2d-1m-3d-fusion-hard20k",
    # The first create request left this slug in a Kaggle ghost state: it is
    # absent from listings/status but create retries return "Notebook not found".
    "multi2d": "nothingnessvoid/molgap-p819-multi2d-1m3d",
}
KERNEL_TITLES = {
    "coverage2m": "MolGap 2M-2D + 1M-3D Fusion coverage2m",
    "hard20k": "MolGap 2M-2D + 1M-3D Fusion hard20k",
    "multi2d": "MolGap P819 Multi2D 1M3D",
}


def _fusion_head_source(path: Path) -> str:
    """Extract the canonical class for Kaggle's single-file script upload."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "FusionHead":
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                return segment
    raise RuntimeError(f"FusionHead was not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo_root = next(
        (parent for parent in root.parents if (parent / "src/molgap/fusion.py").exists()),
        None,
    )
    if repo_root is None:
        raise FileNotFoundError("Could not locate src/molgap/fusion.py from the package")
    fusion_source = repo_root / "src/molgap/fusion.py"
    fusion_head = _fusion_head_source(fusion_source)
    train_source = (root / "train_fusion.py").read_text(encoding="utf-8")
    import_line = "from fusion import FusionHead"
    if train_source.count(import_line) != 1:
        raise RuntimeError("Expected one FusionHead import in train_fusion.py")
    entry_source = train_source.replace(
        import_line,
        "# Injected from src/molgap/fusion.py because Kaggle script kernels "
        "upload only code_file.\n" + fusion_head,
    )
    variant_line = 'VARIANT = json.loads(Path("variant.json").read_text(encoding="utf-8"))["variant"]'
    if entry_source.count(variant_line) != 1:
        raise RuntimeError("Expected one variant.json loader in train_fusion.py")
    for variant in VARIANTS:
        out = args.out_root / variant
        out.mkdir(parents=True, exist_ok=True)
        variant_source = entry_source.replace(variant_line, f'VARIANT = "{variant}"')
        (out / "train_fusion.py").write_text(variant_source, encoding="utf-8")
        (out / "variant.json").write_text(json.dumps({"variant": variant}, indent=2), encoding="utf-8")
        metadata = {
            "id": KERNEL_IDS[variant],
            "title": KERNEL_TITLES[variant],
            "code_file": "train_fusion.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": ["nothingnessvoid/1m-full", args.dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
