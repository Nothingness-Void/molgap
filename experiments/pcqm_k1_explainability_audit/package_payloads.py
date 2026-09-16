"""Package same-contract K1 development payloads for remote analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "platforms/_records/kaggle/training"
OUTPUT = ROOT / "platforms/_records/scnet/k1_explainability_payloads_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    payloads = sorted(RECORDS.rglob("best_development_payload.pt"))
    selected = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source in payloads:
        mode = source.parent.name
        if not mode.startswith("neural_atom_k1") and mode != "neural_atom_k4_cluster":
            continue
        destination = OUTPUT / f"{mode}.pt"
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Conflicting payloads for {mode}")
        shutil.copy2(source, destination)
        selected.append(
            {
                "mode": mode,
                "file": destination.name,
                "sha256": sha256_file(destination),
                "source_record": str(source.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    names = {item["mode"] for item in selected}
    if "neural_atom_k1_v4" not in names or len(selected) < 10:
        raise RuntimeError(f"Incomplete K1 payload set: {sorted(names)}")
    manifest = {
        "format": "molgap-k1-v4-development-payload-set-v1",
        "payloads": selected,
        "payload_count": len(selected),
        "role": "official-train-derived-development",
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    (OUTPUT / "payload_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

