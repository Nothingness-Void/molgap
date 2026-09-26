"""Verified-source bootstrap; training recipe lives in molgap."""
import hashlib
import os
from pathlib import Path
import sys
import tarfile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
matches = list(Path("/kaggle/input").rglob("source_payload.bin"))
if len(matches) != 1:
    raise RuntimeError(f"Expected one immutable source payload, found {len(matches)}")
archive = matches[0]
expected = (archive.parent / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Source payload SHA mismatch")
source = Path("/kaggle/working/verified_relation_source")
source.mkdir(parents=True, exist_ok=False)
with tarfile.open(archive, "r:gz") as stream:
    stream.extractall(source, filter="data")
sys.path.insert(0, str(source / "src"))
from molgap.k1_relation_study_runtime import bootstrap
bootstrap(source, archive, "rrwp")
