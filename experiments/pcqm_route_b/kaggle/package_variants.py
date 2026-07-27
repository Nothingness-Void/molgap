"""Materialize the four self-contained Route B Kaggle kernel packages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
DATASETS = {
    "gps": "nothingnessvoid/molgap-pcqm-route-b-gps-1m-20260727",
    "primary": "nothingnessvoid/molgap-pcqm-route-b-primary-1m-20260727",
    "secondary": "nothingnessvoid/molgap-pcqm-route-b-secondary-1m-20260727",
    "warm": "nothingnessvoid/molgap-pcqm-route-b-warmstarts-20260727",
}
VARIANTS = {
    "gps9": {
        "warm": "gps9_repaired_2m_seed42.pt",
        "datasets": ["gps", "warm"],
    },
    "gps11_160": {
        "warm": "gps11_160_repaired_2m_seed42.pt",
        "datasets": ["gps", "warm"],
    },
    "primary_schnet": {
        "warm": "primary_schnet_100k.pt",
        "datasets": ["primary", "warm"],
    },
    "augmented_schnet": {
        "warm": "augmented_schnet_100k.pt",
        "datasets": ["primary", "secondary", "warm"],
    },
}


def main() -> None:
    template = (HERE / "run_route_b_encoder.py").read_text(encoding="utf-8")
    for name, spec in VARIANTS.items():
        package = HERE / "packages" / name
        package.mkdir(parents=True, exist_ok=True)
        script = template.replace(
            "ENCODER_NAME = None", f'ENCODER_NAME = "{name}"'
        ).replace(
            "WARM_START_NAME = None",
            f'WARM_START_NAME = "{spec["warm"]}"',
        )
        (package / "run_route_b_encoder.py").write_text(
            script, encoding="utf-8"
        )
        for source in (
            ROOT / "src/molgap/gps.py",
            ROOT / "src/molgap/schnet.py",
            ROOT / "src/molgap/pcqm_route_b_training.py",
        ):
            shutil.copy2(source, package / source.name)
        slug = name.replace("_", "-")
        metadata = {
            "id": f"nothingnessvoid/molgap-pcqm-route-b-{slug}-1m-r1-20260727",
            "title": f"MolGap PCQM Route B {name} 1M R1 20260727",
            "code_file": "run_route_b_encoder.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [DATASETS[key] for key in spec["datasets"]],
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
