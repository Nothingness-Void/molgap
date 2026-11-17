from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .training_reproducibility import sha256_file


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def accept_kaggle_runtime(
    *,
    payload_root: Path,
    metadata_path: Path,
    expected_account: str,
    expected_commit: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = payload_root.resolve()
    manifest_path = root / "runtime_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Remote runtime manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-v5-desktop-runtime-v1":
        raise RuntimeError("Unexpected runtime manifest format")
    if manifest.get("status") != "complete":
        raise RuntimeError("Remote runtime is incomplete")
    if manifest.get("source_commit") != expected_commit:
        raise RuntimeError("Remote runtime source commit mismatch")
    if manifest.get("scientific_contract_included") is not False:
        raise RuntimeError("Runtime must not embed a scientific contract")
    if manifest.get("training_authorized") is not False:
        raise RuntimeError("Runtime must not authorize training")

    if (root / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip() != expected_commit:
        raise RuntimeError("Remote SOURCE_COMMIT sidecar mismatch")
    if (
        root.joinpath("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
        != manifest["archive_sha256"]
    ):
        raise RuntimeError("Remote archive SHA256 sidecar mismatch")
    source_inventory_path = root / "SOURCE_FILES.json"
    if sha256_file(source_inventory_path) != manifest["source_files_sha256"]:
        raise RuntimeError("Remote source inventory SHA256 mismatch")
    source_inventory = json.loads(source_inventory_path.read_text(encoding="utf-8"))
    if source_inventory.get("source_commit") != expected_commit:
        raise RuntimeError("Remote source inventory commit mismatch")

    source_root = root / "molgap_runtime"
    expected_paths: set[str] = set()
    payload_bytes = 0
    for item in source_inventory.get("files", []):
        relative = str(item["path"]).replace("\\", "/")
        path = source_root / relative
        if not _inside(path, source_root) or not path.is_file():
            raise RuntimeError(f"Missing or unsafe runtime source path: {relative}")
        if relative in expected_paths:
            raise RuntimeError(f"Duplicate runtime source path: {relative}")
        expected_paths.add(relative)
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Runtime source size mismatch: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Runtime source SHA256 mismatch: {relative}")
        payload_bytes += path.stat().st_size

    observed_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("Remote runtime expanded-file inventory mismatch")
    if len(expected_paths) != int(manifest["file_count"]):
        raise RuntimeError("Remote runtime file count mismatch")
    if payload_bytes != int(manifest["payload_bytes"]):
        raise RuntimeError("Remote runtime payload byte count mismatch")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if metadata.get("ownerUser") != expected_account:
        raise RuntimeError("Remote runtime owner mismatch")
    if metadata.get("datasetSlug") != "molgap-v5-desktop-runtime":
        raise RuntimeError("Remote runtime dataset slug mismatch")
    if metadata.get("isPrivate") is not True:
        raise RuntimeError("Remote runtime must remain private")

    return {
        "status": "accepted",
        "ref": f"{expected_account}/molgap-v5-desktop-runtime",
        "dataset_id": metadata["datasetId"],
        "private": True,
        "source_commit": expected_commit,
        "runtime_manifest_sha256": expected_manifest_sha256,
        "archive_sha256": manifest["archive_sha256"],
        "source_files_sha256": manifest["source_files_sha256"],
        "source_files": len(expected_paths),
        "payload_bytes": payload_bytes,
        "expanded_files_match": True,
        "scientific_contract_included": False,
        "training_authorized": False,
    }
