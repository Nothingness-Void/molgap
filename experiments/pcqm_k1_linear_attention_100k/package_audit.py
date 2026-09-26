"""Package only already accepted frozen candidate artifacts, never new weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import atomic_json, sha256_file
from experiments.pcqm_k1_linear_attention_100k.accept import accept_training

ROOT = REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / "audit_recovery.json").read_text())
    accept_training(args.reference_root, args.candidate_root,
                    spec["source_commit"], spec["source_archive_sha256"])
    if args.output.exists():
        raise FileExistsError("Recovery package must be a new immutable directory")
    for name, digest in spec["candidate_files"].items():
        if sha256_file(args.candidate_root / spec["mode"] / name) != digest:
            raise RuntimeError(f"Accepted recovery input changed: {name}")
    args.output.mkdir(parents=True)
    archive = args.output / "candidate_payload.bin"
    with tarfile.open(archive, "w:gz") as handle:
        for name in spec["candidate_files"]:
            handle.add(args.candidate_root / spec["mode"] / name,
                       arcname=f"{spec['mode']}/{name}")
    spec["candidate_archive_sha256"] = sha256_file(archive)
    spec["launcher_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    spec["launcher_sha256"] = sha256_file(ROOT / "kaggle_audit/run.py")
    atomic_json(args.output / "AUDIT_RECOVERY.json", spec)
    atomic_json(args.output / "dataset-metadata.json", {
        "id": "nothingnessvoid/molgap-k1-linear-attention-audit-inputs",
        "title": "MolGap K1 Linear Attention Frozen Audit Inputs",
        "licenses": [{"name": "CC0-1.0"}],
    })
    print(json.dumps({"package": str(args.output), "archive_sha256": spec["candidate_archive_sha256"]}))


if __name__ == "__main__":
    main()
