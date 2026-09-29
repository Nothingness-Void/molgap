"""Stage one accepted checkpoint dataset and its next Kaggle kernel version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from molgap.constants import REPO_ROOT
from molgap.gptrans_v5_audit_acceptance import RESUME_FILES, accept_segment
from molgap.training_reproducibility import sha256_file


DATASET_ID = "kaseichou/molgap-gptrans-v5-audit-progress-v1"
BASE = Path(REPO_ROOT) / "experiments/pcqm_gptrans_v5_audit_reference/kaggle_t4"


def prepare(root: Path, accepted_epochs: int, progress_output: Path, kernel_output: Path) -> dict:
    if progress_output.exists() or kernel_output.exists():
        raise FileExistsError("Continuation package directory already exists")
    if accepted_epochs not in (10, 20, 30, 40, 50):
        raise ValueError("Only a nonterminal ten-epoch boundary may continue")
    # Revalidate the retained raw bytes before a new GPU job can use them.
    report = accept_segment(root, accepted_epochs, root / "segment_acceptance.json")
    if not report["next_segment_authorized"]:
        raise RuntimeError("Segment acceptance did not authorize continuation")
    progress_output.mkdir(parents=True)
    training = root / "training"
    for name in (*RESUME_FILES, "partial_manifest.json"):
        shutil.copyfile(training / name, progress_output / name)
    (progress_output / "dataset-metadata.json").write_text(json.dumps({
        "title": "MolGap GPTrans V5 Audit Progress V1",
        "id": DATASET_ID, "licenses": [{"name": "other"}], "isPrivate": True,
    }, indent=2), encoding="utf-8")
    kernel_output.mkdir(parents=True)
    script = (BASE / "run.py").read_text(encoding="utf-8")
    marker = "EXPECTED_PREVIOUS_EPOCHS = None"
    if script.count(marker) != 1:
        raise RuntimeError("Continuation marker changed")
    (kernel_output / "run.py").write_text(
        script.replace(marker, f"EXPECTED_PREVIOUS_EPOCHS = {accepted_epochs}"),
        encoding="utf-8",
    )
    metadata = json.loads((BASE / "kernel-metadata.json").read_text(encoding="utf-8"))
    if DATASET_ID in metadata["dataset_sources"]:
        raise RuntimeError("Progress dataset already present in template")
    metadata["dataset_sources"].append(DATASET_ID)
    (kernel_output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "accepted_epochs": accepted_epochs,
        "source_checkpoint_sha256": report["checkpoint_sha256"],
        "progress_dataset_id": DATASET_ID,
        "kernel_id": metadata["id"],
        "kernel_run_sha256": sha256_file(kernel_output / "run.py"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--progress-output", required=True, type=Path)
    parser.add_argument("--kernel-output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.epochs, args.progress_output, args.kernel_output), sort_keys=True))


if __name__ == "__main__":
    main()
