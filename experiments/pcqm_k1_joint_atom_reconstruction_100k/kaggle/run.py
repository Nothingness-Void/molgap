"""Immutable-source bootstrap; scientific execution stays in molgap."""
import hashlib
import os
from pathlib import Path
import sys
import tarfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
matches = list(Path("/kaggle/input").rglob("source_payload.bin"))
if len(matches) != 1:
    raise RuntimeError("Expected one immutable source payload")
archive = matches[0]
digest = (archive.parent / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
    raise RuntimeError("Source archive SHA mismatch")
source = Path("/kaggle/working/verified_joint_source")
source.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive, "r:gz") as stream:
    stream.extractall(source, filter="data")
sys.path.insert(0, str(source / "src"))
from molgap.k1_joint_study_runtime import verify_source, run
run(verify_source(source, archive))
