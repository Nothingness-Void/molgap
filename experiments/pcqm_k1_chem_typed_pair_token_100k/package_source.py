"""Package the exact committed source tree bound by the V5 prelaunch."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT / "experiments/pcqm_k1_chem_typed_pair_token_100k"


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="kaseichou/molgap-k1-chem-typed-pair-token-source"
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "source_config.json").read_text(encoding="utf-8"))
    commit = config["source_commit"]
    subprocess.check_call(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO_ROOT)
    raw = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", commit, "--", "src"],
        cwd=REPO_ROOT,
    )
    names = sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    inventory = []
    archive_path = output / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in names:
            if "__pycache__" in Path(name).parts:
                continue
            data = git_bytes(commit, name)
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, BytesIO(data))
            inventory.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
            expanded = output / name
            expanded.parent.mkdir(parents=True, exist_ok=True)
            expanded.write_bytes(data)
    payload = archive_path.read_bytes()
    (output / "source_payload.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": args.dataset_id,
                "title": "MolGap K1 Chemistry Typed PairToken Source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest}))


if __name__ == "__main__":
    main()
