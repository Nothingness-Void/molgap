"""Package only the two frozen checkpoints and their target normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED = {
    "full_gps_best.pt": "467753c8caa26e3e8d537aa7933e693073fcc45851caf561174d604742d48e6a",
    "neural_atom_k1_best.pt": "9e9ac63a9887030dcfbdc795d843cc10e70f8d9dcec7421a13cff98970203784",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--target-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="kaseichou/molgap-pcqm-k1-shadow-checkpoints"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    sources = {
        "full_gps_best.pt": args.records_root / "full_gps" / "best_model.pt",
        "neural_atom_k1_best.pt": args.records_root
        / "neural_atom_k1"
        / "best_model.pt",
    }
    for name, source in sources.items():
        if sha256_file(source) != EXPECTED[name]:
            raise RuntimeError(f"Frozen checkpoint changed: {source}")
        shutil.copy2(source, args.output / name)
    shutil.copy2(args.target_stats, args.output / "target_stats.json")
    manifest = {
        "format": "molgap-pcqm-k1-shadow-checkpoints-v1",
        "source_training_task": "pcqm-gap100k-fourier-edge-s42-v1",
        "frozen_model_source_commit": "47f99cf9da7fee306f5165175b4020c6c4aa9fb3",
        "checkpoint_sha256": EXPECTED,
        "target_stats_sha256": sha256_file(args.output / "target_stats.json"),
        "shadow_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "model_inference_executed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap PCQM K1 Shadow Checkpoints",
                "id": args.dataset_id,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
