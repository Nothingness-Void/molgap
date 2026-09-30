"""Hash-pinned bootstrap; scheduling and training remain in shared owners."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

SOURCE_SHA256 = "__PIN_SOURCE_ARCHIVE_SHA256__"
OUTPUT = Path("/kaggle/working/gptrans_author_screen")


def main():
    matches = [p for p in Path("/kaggle/input").rglob("source_payload.bin")
               if hashlib.sha256(p.read_bytes()).hexdigest() == SOURCE_SHA256]
    if len(matches) != 1:
        raise RuntimeError("Missing or ambiguous frozen GPU source archive")
    archive = matches[0]
    inventory = json.loads((archive.parent / "SOURCE_FILES.json").read_text())
    entries = {row["path"]: row for row in inventory["files"]}
    source = OUTPUT / "verified_source"
    source.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or member.name not in entries or member.name in seen):
                raise RuntimeError("Unsafe archive member")
            payload = bundle.extractfile(member).read()
            row = entries[member.name]
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError("Source member differs from the frozen inventory")
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
            seen.add(member.name)
    if seen != set(entries):
        raise RuntimeError("Incomplete executable source archive")
    sys.path.insert(0, str(source / "src"))
    from molgap.gptrans_author_screen import run_author_screen
    run_author_screen(archive.parent, source, OUTPUT, SOURCE_SHA256)


if __name__ == "__main__":
    main()
