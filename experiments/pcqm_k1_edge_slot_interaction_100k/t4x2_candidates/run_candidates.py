"""Verify committed source, then launch the frozen two-arm runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

MODES = ["neural_atom_k1_edge_context_no_slot_attention", "neural_atom_k1_edge_context_uniform_return"]


def find_one(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def main():
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    modules = list(Path("/kaggle/input").rglob("src/molgap/k1_edge_slot_interaction.py"))
    if len(modules) == 1:
        root = modules[0].parents[1]
    else:
        expanded = Path("/kaggle/working/_k1_edge_slot_source")
        shutil.unpack_archive(archive, expanded, format="gztar")
        root = expanded / "src"
    for item in json.loads(find_one("SOURCE_FILES.json").read_text())["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "src":
            raise RuntimeError("Invalid source inventory path")
        mounted = root.joinpath(*relative.parts[1:])
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {relative}")
    sys.path.insert(0, str(root))
    os.environ["MOLGAP_SCREEN_MODES"] = json.dumps(MODES)
    from molgap.k1_edge_kaggle_runtime import main as run
    run({"python_root": str(root), "source_commit": commit, "source_archive_sha256": digest})


if __name__ == "__main__":
    main()
