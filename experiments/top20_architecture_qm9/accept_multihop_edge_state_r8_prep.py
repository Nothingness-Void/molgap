"""Accept downloaded R8 multihop cache parts without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def aggregate_sha256(parts: list[dict]) -> str:
    identity = [
        {"name": row["name"], "rows": row["rows"], "sha256": row["sha256"]}
        for row in parts
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prep_path = find_one(root, "prep_result.json")
    acceptance_path = find_one(root, "multihop_acceptance.json")
    progress_path = find_one(root, "progress.json")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected = {
        "format": "molgap-qm9-multihop-acceptance-v1",
        "complete": True,
        "split_seed": 42,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "roles": {"train": 30_000, "validation": 3_000},
        "requested_test_rows": 3_000,
        "test_role_read": False,
        "max_distance": 4,
        "bond_feature_dim": 4,
        "distance_feature_dim": 4,
        "edge_feature_dim": 8,
        "rows": 33_000,
    }
    mismatches = {
        key: (acceptance.get(key), value)
        for key, value in expected.items()
        if acceptance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"R8 cache contract mismatch: {mismatches}")
    if prep.get("format") != "molgap-multihop-edgestate-r8-prep-result-v1":
        raise RuntimeError("Unexpected R8 prep result format")
    if prep.get("source_commit") != args.source_commit:
        raise RuntimeError("R8 source commit mismatch")
    if prep.get("gpu_used") is not False or prep.get("test_role_read") is not False:
        raise RuntimeError("R8 prep violated its resource or role boundary")
    if prep.get("acceptance") != acceptance:
        raise RuntimeError("Prep result embeds a different acceptance record")
    parts = acceptance.get("parts", [])
    if not parts or sum(int(row["rows"]) for row in parts) != 33_000:
        raise RuntimeError("R8 cache part rows are incomplete")
    if acceptance.get("parts_sha256") != aggregate_sha256(parts):
        raise RuntimeError("R8 cache aggregate hash mismatch")
    if progress.get("complete_parts") != len(parts):
        raise RuntimeError("R8 cache progress is incomplete")
    part_root = acceptance_path.parent / "parts"
    accepted_parts = []
    for part in parts:
        path = part_root / part["name"]
        if path.name != part["name"] or not path.is_file():
            raise RuntimeError(f"Missing R8 cache part: {part['name']}")
        observed = sha256(path)
        if observed != part["sha256"] or path.stat().st_size != part["bytes"]:
            raise RuntimeError(f"R8 cache part identity mismatch: {part['name']}")
        accepted_parts.append(
            {
                "name": part["name"],
                "rows": part["rows"],
                "bytes": part["bytes"],
                "sha256": observed,
            }
        )
    report = {
        "format": "molgap-multihop-edgestate-r8-prep-local-acceptance-v1",
        "accepted": True,
        "source_commit": args.source_commit,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "test_role_read": False,
        "model_inference_executed": False,
        "parts_sha256": acceptance["parts_sha256"],
        "rows": acceptance["rows"],
        "edge_rows": acceptance["edge_rows"],
        "parts": accepted_parts,
        "prep_result_sha256": sha256(prep_path),
        "acceptance_sha256": sha256(acceptance_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"accepted": True, "parts": len(parts)}))


if __name__ == "__main__":
    main()
