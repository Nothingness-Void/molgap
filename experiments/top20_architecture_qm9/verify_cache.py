"""Verify the immutable ETKDG manifest before any GPU training starts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError("ETKDG acceptance manifest is not complete")
    files = [payload["cache"], payload["report_file"], *payload["independent_shards"]]
    checked = []
    for item in files:
        path = Path(item["path"])
        if not path.is_file():
            raise RuntimeError(f"Accepted cache file is missing: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Accepted cache hash changed: {path}")
        checked.append({"path": str(path), "sha256": actual})
    result = {
        "format": "molgap-top20-qm9-etkdg-verified-v1",
        "status": "complete",
        "source_manifest": str(args.manifest),
        "files": checked,
        "rows": payload["report"],
    }
    args.verified_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.verified_output.with_suffix(args.verified_output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.verified_output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
