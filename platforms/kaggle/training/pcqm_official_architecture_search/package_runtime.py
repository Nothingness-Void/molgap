"""Package the exact MolGap modules required by the Kaggle architecture screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODULES = (
    "gps.py",
    "ogb_features.py",
    "pair_gps_2d.py",
    "pcqm_feature_screen.py",
    "pcqm_official_edge_state.py",
    "structural_encoding.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.repo_root.resolve() / "src" / "molgap"
    output = args.output.resolve()
    package = output / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    hashes = {}
    minimal_init = package / "__init__.py"
    minimal_init.write_text(
        '"""Minimal runtime package for bounded PCQM architecture screens."""\n',
        encoding="utf-8",
    )
    hashes["molgap/__init__.py"] = sha256(minimal_init)
    for name in MODULES:
        source_path = source / name
        destination = package / name
        shutil.copy2(source_path, destination)
        hashes[f"molgap/{name}"] = sha256(destination)
    (output / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "format": "molgap-pcqm-architecture-runtime-v3",
                "modules": hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap PCQM Architecture Runtime V8 20260828",
                "id": "nothingnessvoid/molgap-pcqm-architecture-runtime-v8-20260828",
                "licenses": [{"name": "CC0-1.0"}],
                "description": (
                    "Private official-train-only runtime for bounded PCQM "
                    "architecture screens; no model weights or labels."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
