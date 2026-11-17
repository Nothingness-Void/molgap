from __future__ import annotations

import hashlib
import csv
import json
import os
from pathlib import Path
from typing import Any


FIXED_MIRRORS: dict[str, dict[str, Any]] = {
    "100k": {
        "slug": "pcqm4mv2-ogb-fixed-100k-v1",
        "title": "PCQM4Mv2 OGB Fixed 100K V1",
        "manifest_sha256": (
            "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
        ),
        "identity_name": "ogb-train-100k",
        "train_rows": 100_000,
        "development_rows": 50_000,
        "graph_shards": 3,
    },
    "500k": {
        "slug": "pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "title": "PCQM4Mv2 OGB Fixed 500K SCNet V1",
        "manifest_sha256": (
            "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
        ),
        "identity_name": "ogb-train-500k-scnet-v1",
        "train_rows": 500_000,
        "development_rows": 50_000,
        "graph_shards": 11,
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_fixed_mirror(payload_root: Path, role: str) -> dict[str, Any]:
    root = payload_root.resolve()
    expected = FIXED_MIRRORS[role]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected["manifest_sha256"]:
        raise RuntimeError(
            f"Manifest identity mismatch: {manifest_sha256} != "
            f"{expected['manifest_sha256']}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
        raise RuntimeError("Unexpected fixed-dataset manifest format")
    if manifest.get("status") != "complete":
        raise RuntimeError("Fixed-dataset manifest is incomplete")

    identity = manifest.get("identity", {})
    if identity.get("name") != expected["identity_name"]:
        raise RuntimeError("Fixed-dataset identity name mismatch")
    if identity.get("train_rows") != expected["train_rows"]:
        raise RuntimeError("Fixed-dataset train row count mismatch")
    if identity.get("development_rows") != expected["development_rows"]:
        raise RuntimeError("Fixed-dataset development row count mismatch")

    protected_flags = (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    )
    if any(manifest.get(flag) is not False for flag in protected_flags):
        raise RuntimeError("A protected PCQM4Mv2 role is present or ambiguous")

    shards = manifest.get("geometry_shards", [])
    if len(shards) != expected["graph_shards"]:
        raise RuntimeError("Fixed-dataset graph shard count mismatch")

    declared_paths: set[str] = set()
    payload_bytes = 0
    payload_rows = 0
    for shard in shards:
        relative = str(shard["file"]).replace("\\", "/")
        path = root / relative
        if not _inside(path, root) or not path.is_file():
            raise RuntimeError(f"Invalid graph shard path: {relative}")
        if relative in declared_paths:
            raise RuntimeError(f"Duplicate graph shard path: {relative}")
        declared_paths.add(relative)
        if path.stat().st_size != int(shard["bytes"]):
            raise RuntimeError(f"Graph shard size mismatch: {relative}")
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Graph shard SHA256 mismatch: {relative}")
        payload_bytes += path.stat().st_size
        payload_rows += int(shard["rows"])

    undeclared_graphs = {
        path.relative_to(root).as_posix() for path in root.rglob("*.pt")
    } - declared_paths
    if undeclared_graphs:
        raise RuntimeError(
            "Undeclared graph shards: " + ", ".join(sorted(undeclared_graphs))
        )

    return {
        "status": "accepted_for_mirror",
        "role": role,
        "manifest_sha256": manifest_sha256,
        "graph_shards": len(shards),
        "payload_rows": payload_rows,
        "payload_bytes": payload_bytes,
        "protected_roles_included": False,
    }


def prepare_account_mirror(
    payload_root: Path, account: str, role: str
) -> dict[str, Any]:
    if not account or "/" in account or "\\" in account:
        raise ValueError("Kaggle account must be a single non-empty slug")
    report = verify_fixed_mirror(payload_root, role)
    expected = FIXED_MIRRORS[role]
    metadata = {
        "id": f"{account}/{expected['slug']}",
        "title": expected["title"],
        "licenses": [{"name": "CC-BY-4.0"}],
        "isPrivate": True,
    }
    destination = payload_root.resolve() / "dataset-metadata.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return {**report, "dataset_ref": metadata["id"]}


def verify_published_mirror(
    *,
    manifest_path: Path,
    remote_files_path: Path,
    remote_metadata_path: Path,
    account: str,
    role: str,
) -> dict[str, Any]:
    expected = FIXED_MIRRORS[role]
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected["manifest_sha256"]:
        raise RuntimeError("Published manifest SHA256 does not match the fixed identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metadata = json.loads(remote_metadata_path.read_text(encoding="utf-8-sig"))
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if metadata.get("ownerUser") != account:
        raise RuntimeError("Published dataset owner mismatch")
    if metadata.get("datasetSlug") != expected["slug"]:
        raise RuntimeError("Published dataset slug mismatch")
    if metadata.get("isPrivate") is not True:
        raise RuntimeError("Published fixed dataset must remain private")
    if not isinstance(metadata.get("datasetId"), int):
        raise RuntimeError("Published dataset lacks a numeric dataset id")

    with remote_files_path.open("r", encoding="utf-8-sig", newline="") as handle:
        remote_rows = list(csv.DictReader(line for line in handle if line.strip()))
    remote_files = {row["name"]: int(row["size"]) for row in remote_rows}
    expected_files = {
        "README.md": 490,
        "manifest.json": manifest_path.stat().st_size,
        **{
            str(shard["file"]).replace("\\", "/"): int(shard["bytes"])
            for shard in manifest["geometry_shards"]
        },
    }
    if remote_files != expected_files:
        missing = sorted(set(expected_files) - set(remote_files))
        extra = sorted(set(remote_files) - set(expected_files))
        wrong_size = sorted(
            name
            for name in set(expected_files) & set(remote_files)
            if expected_files[name] != remote_files[name]
        )
        raise RuntimeError(
            "Published file inventory mismatch: "
            f"missing={missing}, extra={extra}, wrong_size={wrong_size}"
        )

    return {
        "status": "accepted",
        "role": role,
        "ref": f"{account}/{expected['slug']}",
        "dataset_id": metadata["datasetId"],
        "private": True,
        "manifest_sha256": manifest_sha256,
        "graph_shards": len(manifest["geometry_shards"]),
        "remote_files": len(remote_files),
        "remote_file_inventory_matches": True,
        "protected_roles_included": False,
    }
