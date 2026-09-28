"""Publish only committed K1 source after both real prelaunch gates pass."""
import argparse
import json
from pathlib import Path
import sys

from molgap.constants import REPO_ROOT
from molgap.k1_chem_local import MODES
from molgap.k1_chem_local_study_runtime import RUN_ID, TRAJECTORIES
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.server_acceptance import validate_server_scientific_prelaunch
from experiments.pcqm_k1_sparse_triplet_100k import package_source as common


REL = "experiments/pcqm_k1_chem_local_100k"
REFERENCE = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"
DATASET_ID = "kaseichou/molgap-k1-chem-local-source"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args()
    if args.dataset_id != DATASET_ID:
        raise ValueError("Source dataset owner/identity differs from frozen experiment")
    root = REPO_ROOT / REL
    bundle = json.loads((REPO_ROOT / REFERENCE).read_text())
    frozen = json.loads((root / "source_config.json").read_text())
    release = {"source_commit": frozen["source_commit"], "run_id": RUN_ID,
        "arms": {}, "reference_bundle_sha256": file_digest(REPO_ROOT / REFERENCE)}
    for mode in MODES:
        arm = mode.removeprefix("neural_atom_k1_")
        path = root / "arms" / arm / "comparison_readiness_prelaunch.json"
        prelaunch = json.loads(path.read_text())
        validate_server_scientific_prelaunch(
            comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
            reference_bundle=bundle, repo_root=REPO_ROOT,
            reference_bundle_path=REPO_ROOT / REFERENCE)
        release["arms"][mode] = {"prelaunch_ready": prelaunch["prelaunch_ready"],
            "prelaunch_sha256": file_digest(path), "trajectory_id": TRAJECTORIES[mode]}
    common.EXPERIMENT = REL
    previous = sys.argv
    try:
        sys.argv = [previous[0], "--output", str(args.output), "--dataset-id", DATASET_ID]
        common.main()
    finally:
        sys.argv = previous
    atomic_write(args.output / "STUDY_RELEASE.json", json_bytes(release))
    metadata = json.loads((args.output / "dataset-metadata.json").read_text())
    metadata["title"] = "MolGap K1 Chemistry-Separated Local Source"
    atomic_write(args.output / "dataset-metadata.json", json_bytes(metadata))


if __name__ == "__main__":
    main()
