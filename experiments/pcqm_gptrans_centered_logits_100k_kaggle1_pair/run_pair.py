"""Extract the frozen source and select the existing GPTrans T4x2 runner."""
from __future__ import annotations

import hashlib
import os
import runpy
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


archive = one("source_payload.bin")
expected = one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
with archive.open("rb") as handle:
    observed = hashlib.file_digest(handle, "sha256").hexdigest()
if observed != expected:
    raise RuntimeError("Frozen source archive hash changed")
root = Path("/kaggle/temp/molgap-source")
root.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle:
        name = PurePosixPath(member.name)
        if (not member.isfile() or name.is_absolute()
                or any(part in {"", ".", ".."} for part in name.parts)):
            raise RuntimeError("Unsafe source archive member")
        destination = root.joinpath(*name.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stream = bundle.extractfile(member)
        if stream is None:
            raise RuntimeError("Missing source archive member")
        with stream, destination.open("xb") as target:
            shutil.copyfileobj(stream, target)

os.environ["MOLGAP_PAIR_PROFILE"] = "reference-centered-logits-kaggle1-v1"
os.environ["MOLGAP_SOURCE_ROOT"] = str(root)
runpy.run_path(
    str(root / "experiments/pcqm_gptrans_pair_norm_100k/run_candidates.py"),
    run_name="__main__",
)
