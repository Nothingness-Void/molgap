"""Package only committed source after each real reference release gate."""
import argparse
import json
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.server_acceptance import validate_server_scientific_prelaunch
from molgap.research_memory.trace import file_digest, json_bytes, atomic_write
from molgap.k1_relation_study_runtime import SLOTS, RUNS, TRAJECTORIES
from experiments.pcqm_k1_sparse_triplet_100k import package_source as common

REL = "experiments/pcqm_k1_relation_resolution_100k"
REFERENCE = "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default="kaseichou/molgap-k1-relation-resolution-source")
    args = parser.parse_args()
    if args.dataset_id != "kaseichou/molgap-k1-relation-resolution-source":
        raise ValueError("Owner/source identity differs from the frozen experiment")
    root = REPO_ROOT / REL
    bundle = json.loads((REPO_ROOT / REFERENCE).read_text())
    release = {"source_commit": json.loads((root / "source_config.json").read_text())["source_commit"],
               "slot_runs": RUNS, "arms": {}, "reference_bundle_sha256": file_digest(REPO_ROOT / REFERENCE)}
    for mode in TRAJECTORIES:
        arm = mode.removeprefix("neural_atom_k1_")
        prelaunch_path = root / "arms" / arm / "comparison_readiness_prelaunch.json"
        prelaunch = json.loads(prelaunch_path.read_text())
        validate_server_scientific_prelaunch(
            comparison_prelaunch=prelaunch, experiment_purpose="architecture_comparison",
            reference_bundle=bundle, repo_root=REPO_ROOT,
            reference_bundle_path=REPO_ROOT / REFERENCE)
        release["arms"][mode] = {"prelaunch_ready": prelaunch["prelaunch_ready"],
            "prelaunch_sha256": file_digest(prelaunch_path), "trajectory_id": TRAJECTORIES[mode]}
    common.EXPERIMENT = REL
    common.main()
    atomic_write(args.output / "STUDY_RELEASE.json", json_bytes(release))
    metadata_path = args.output / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["title"] = "MolGap K1 Relation Resolution Source"
    atomic_write(metadata_path, json_bytes(metadata))


if __name__ == "__main__":
    main()
