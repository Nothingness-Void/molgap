"""Package the committed MetaGIN source with the shared V4 archive builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import sha256_file
from molgap.v4_bundle import build_v4_source_bundle


DATASET_ID = "kaseichou/molgap-metagin-2d-source"
TRANSFORM_REF = (
    "experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/target_transform.json"
)
TRANSFORM_SHA = "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src/molgap"], cwd=REPO_ROOT,
    ).split(b"\0")
    names = [part.decode("utf-8") for part in tracked if part and part.endswith(b".py")]
    bundle = build_v4_source_bundle(
        repo_root=REPO_ROOT, relative_paths=names,
        output_dir=output, source_commit=commit,
    )
    shutil.copyfile(output / "source.tar.gz", output / "source_payload.bin")
    asset = REPO_ROOT / TRANSFORM_REF
    if sha256_file(asset) != TRANSFORM_SHA:
        raise RuntimeError("K1 target-transform asset changed")
    shutil.copyfile(asset, output / "target_transform.json")
    (output / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET_ID, "title": "MolGap MetaGIN2D V5 Source",
        "licenses": [{"name": "other"}], "isPrivate": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_commit": commit,
        "source_archive_sha256": bundle["archive_sha256"],
        "source_file_count": bundle["file_count"],
        "target_transform_file_sha256": TRANSFORM_SHA,
    }, indent=2))


if __name__ == "__main__":
    main()
