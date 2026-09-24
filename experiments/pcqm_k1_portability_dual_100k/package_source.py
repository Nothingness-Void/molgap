"""Publish immutable source plus the accepted K1 reference audit inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from molgap.constants import REPO_ROOT
from experiments.pcqm_k1_sparse_triplet_100k import package_source as common


EXPERIMENT = "experiments/pcqm_k1_portability_dual_100k"
REFERENCE = (
    REPO_ROOT / "platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2"
    / "pcqm_k1_v4_reference/neural_atom_k1_v4"
)
EXPECTED = {
    "K1_REFERENCE_BEST_MODEL.pt": (
        REFERENCE / "best_model.pt",
        "53f9118f34a95e02e3f0d798a56389af53ff55739753b4208c9f53c36d116d95",
    ),
    "K1_REFERENCE_DEVELOPMENT.pt": (
        REFERENCE / "best_development_payload.pt",
        "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91",
    ),
    "K1_REFERENCE_TARGET_TRANSFORM.json": (
        REPO_ROOT / "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json",
        "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="nothingnessvoid/molgap-k1-topology-portability-source"
    )
    args = parser.parse_args()
    common.EXPERIMENT = EXPERIMENT
    common.main()
    output = args.output.resolve()
    for name, (source, digest) in EXPECTED.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen K1 reference bytes changed: {name}")
        shutil.copyfile(source, output / name)
    metadata = json.loads((output / "dataset-metadata.json").read_text(encoding="utf-8"))
    metadata["title"] = "MolGap K1 Topology Portability Source"
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_dataset": args.dataset_id, "reference_hashes": {
        name: digest for name, (_, digest) in EXPECTED.items()
    }}))


if __name__ == "__main__":
    main()
