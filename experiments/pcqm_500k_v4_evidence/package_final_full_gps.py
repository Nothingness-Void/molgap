"""Package the single missing EdgeState epoch from accepted stage-5 output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT
EXPERIMENT = ROOT / "experiments" / "pcqm_500k_v4_evidence"
STAGING = ROOT / "platforms" / "_records" / "kaggle" / "staging" / "pcqm_500k_v4_evidence"
INPUT = (
    ROOT
    / "platforms"
    / "_records"
    / "kaggle"
    / "training"
    / "pcqm_500k_v4_stage5"
    / "edge-k1-v6"
    / "evidence"
    / "full_gps"
)


def main() -> None:
    manifest = json.loads((INPUT / "stage_manifest.json").read_text())
    if manifest["arm"] != "full_gps" or manifest["next_epoch"] != 59:
        raise RuntimeError("Expected accepted full_gps epoch-59 cursor")
    if manifest["next_epoch"] >= 60:
        raise RuntimeError("Refusing to package an already complete arm")

    resume = STAGING / "resume_stage5_full_gps_e59"
    resume.mkdir(parents=True, exist_ok=True)
    archive_path = resume / "stage_resume.bin"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(manifest["artifacts"]) | {"stage_manifest.json"}):
            archive.write(INPUT / name, "full_gps/" + name)
    resume_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    resume_slug = "nothingnessvoid/molgap-500k-v4-full-gps-e59-" + resume_sha[:10]
    (resume / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": resume_slug,
                "title": "MolGap 500K V4 Full GPS Epoch59 " + resume_sha[:10],
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )

    previous = STAGING / "edge-k1-stage5" / "run.py"
    text = previous.read_text()
    replacements = {
        "ARMS = ['full_gps', 'neural_atom_k1']": "ARMS = ['full_gps']",
        "RESUME_SHA = '248f77598300a22736099270ffcb8754627c78cd7c2784424f976efea97c897f'": f"RESUME_SHA = {resume_sha!r}",
        "RESUME_EPOCH = 16": "RESUME_EPOCH = 59",
        "STAGE_EPOCHS = 44": "STAGE_EPOCHS = 1",
        "MAX_STAGE_SECONDS = 41400": "MAX_STAGE_SECONDS = 10800",
        "RESUME_SOURCE_SHA = '811a40969ece2df05d95556901fcc7c3ebcaa6f46c95e19a4e9fa2e856fad3ae'": "RESUME_SOURCE_SHA = None",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f"Frozen wrapper identity changed: {old}")
        text = text.replace(old, new)

    package = STAGING / "full-gps-final-e59"
    package.mkdir(parents=True, exist_ok=True)
    (package / "run.py").write_text(text)
    metadata = json.loads((STAGING / "edge-k1-stage5" / "kernel-metadata.json").read_text())
    metadata["id"] = "nothingnessvoid/molgap-500k-v4-full-gps-final-epoch"
    metadata["title"] = "MolGap 500K V4 Full GPS Final Epoch"
    metadata["dataset_sources"] = [
        "nothingnessvoid/molgap-500k-v4-source-raw-4218940805",
        "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1",
        resume_slug,
    ]
    (package / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    record = {
        "arm": "full_gps",
        "resume_epoch": 59,
        "target_epoch": 60,
        "resume_dataset": resume_slug,
        "resume_archive_sha256": resume_sha,
        "resume_checkpoint_source_sha256": "4218940805a84dbf69971452e3c113c7f5e8281df8be62371836b8bc3ff2b649",
        "kernel": "nothingnessvoid/molgap-500k-v4-full-gps-final-epoch",
        "kernel_version": 8,
        "status_at_submission_check": "RUNNING",
        "predecessor": {
            "kernel_version": 7,
            "terminal_status": "ERROR",
            "error": "Resume scientific/source identity mismatch",
            "training_epochs_completed": 0,
        },
        "package": str(package),
    }
    (EXPERIMENT / "stage6_full_gps_submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
