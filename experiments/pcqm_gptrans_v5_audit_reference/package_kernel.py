"""Stage a self-contained Kaggle kernel from the frozen scientific source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from molgap.constants import REPO_ROOT


SOURCE_SHA256 = "af94a63c3c4626ceec8af4106aad0ea97d398e40aefb04c51b15356b994137a3"
SOURCE_COMMIT = "21f70ab93c3b0ddc4745316daf8459df40efdb48"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    base = Path(REPO_ROOT) / "experiments/pcqm_gptrans_v5_audit_reference/kaggle_t4"
    source = Path(REPO_ROOT) / "platforms/_records/kaggle/packages/gptrans_v5_audit_source_21f70ab9"
    if hashlib.sha256((source / "source_payload.bin").read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Frozen source archive changed")
    if (source / "SOURCE_COMMIT.txt").read_text(encoding="ascii").strip() != SOURCE_COMMIT:
        raise RuntimeError("Frozen source commit changed")
    output.mkdir(parents=True)
    for name in ("run.py", "kernel-metadata.json"):
        shutil.copyfile(base / name, output / name)
    for name in ("source_payload.bin", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(source / name, output / name)
    print(f"Staged verified Kaggle kernel: {output}")


if __name__ == "__main__":
    main()
