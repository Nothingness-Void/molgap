"""Wrap the shared committed-source packager with the real V5 release gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from molgap.constants import REPO_ROOT
from molgap.k1_motif_hierarchy import MODE
from molgap.k1_motif_study_runtime import RUN_ID, TRAJECTORY_ID
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.server_acceptance import validate_server_scientific_prelaunch
from experiments.pcqm_k1_sparse_triplet_100k import package_source as common


REL = "experiments/pcqm_motif_hierarchy_100k"
REFERENCE = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
DATASET_ID = "kaseichou/molgap-k1-motif-hierarchy-source"
ATTEMPT = RUN_ID.rsplit(":", 1)[1].replace("v", "attempt_v", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = REPO_ROOT / REL
    metadata = json.loads((root / "kaggle_gpu/kernel-metadata.json").read_text())
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    if (metadata["id"] != f"kaseichou/{title_slug}"
            or RUN_ID.partition(":")[0] != metadata["id"]
            or set(metadata["dataset_sources"]) != {
                DATASET_ID, "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
                "kaseichou/molgap-motif-partition-cache-v1"}):
        raise ValueError("Frozen Kaggle slug or input mounts changed")
    reference_path = REPO_ROOT / REFERENCE
    bundle = json.loads(reference_path.read_text())
    frozen = json.loads((root / ATTEMPT / "source_config.json").read_text())
    prelaunch_path = root / ATTEMPT / "comparison_readiness_prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text())
    validate_server_scientific_prelaunch(
        comparison_prelaunch=prelaunch,
        experiment_purpose="architecture_comparison", reference_bundle=bundle,
        repo_root=REPO_ROOT, reference_bundle_path=reference_path,
    )
    release = {
        "source_commit": frozen["source_commit"], "run_id": RUN_ID,
        "mode": MODE, "trajectory_id": TRAJECTORY_ID,
        "prelaunch_ready": prelaunch["prelaunch_ready"],
        "prelaunch_sha256": file_digest(prelaunch_path),
        "reference_bundle_sha256": file_digest(reference_path),
        "motif_sidecar_aggregate_sha256": frozen["motif_aggregate_sha256"],
    }
    common.EXPERIMENT = f"{REL}/{ATTEMPT}"
    previous = sys.argv
    try:
        sys.argv = [previous[0], "--output", str(args.output), "--dataset-id", DATASET_ID]
        common.main()
    finally:
        sys.argv = previous
    atomic_write(args.output / "STUDY_RELEASE.json", json_bytes(release))
    packaged = json.loads((args.output / "dataset-metadata.json").read_text())
    packaged["title"] = "MolGap K1 Motif Hierarchy Source"
    atomic_write(args.output / "dataset-metadata.json", json_bytes(packaged))


if __name__ == "__main__":
    main()
