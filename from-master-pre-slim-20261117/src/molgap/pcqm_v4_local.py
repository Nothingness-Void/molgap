"""Local acquisition and integrity checks for the accepted PCQM 500K V4 cache."""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .training_reproducibility import atomic_json, sha256_file


DATASET_SLUG = "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1"
EXPECTED_ROWS = {"train": 500_000, "development": 50_000}


def _dataset_file_url(file_name: str) -> str:
    owner, dataset = DATASET_SLUG.split("/", 1)
    encoded = urllib.parse.quote(file_name, safe="")
    return f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}/{encoded}"


def _authorization(username: str, key: str) -> str:
    payload = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    return f"Basic {payload}"


def _download_file(
    *,
    file_name: str,
    destination: Path,
    expected_sha256: str,
    username: str,
    key: str,
) -> dict:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return {"file": file_name, "status": "reused", "sha256": expected_sha256}
    if destination.exists():
        raise RuntimeError(f"Existing local V4 asset has the wrong identity: {destination}")

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Authorization": _authorization(username, key)}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(_dataset_file_url(file_name), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 << 20)
    observed = sha256_file(partial)
    if observed != expected_sha256:
        raise RuntimeError(
            f"Downloaded V4 asset hash mismatch: {file_name} {observed}"
        )
    os.replace(partial, destination)
    return {"file": file_name, "status": "downloaded", "sha256": observed}


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("format") != "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
        raise RuntimeError("Unexpected fixed-500K manifest format")
    if manifest.get("status") != "complete":
        raise RuntimeError("Fixed-500K manifest is incomplete")
    identity = manifest.get("identity", {})
    if identity.get("name") != "ogb-train-500k-scnet-v1":
        raise RuntimeError("Wrong fixed-500K dataset identity")
    if identity.get("train_rows") != EXPECTED_ROWS["train"]:
        raise RuntimeError("Wrong fixed-500K train count")
    if identity.get("development_rows") != EXPECTED_ROWS["development"]:
        raise RuntimeError("Wrong fixed-500K development count")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Sealed PCQM role was accessed: {role}")


def download_fixed_500k_cache(
    output_root: Path,
    *,
    username: str,
    key: str,
    workers: int = 4,
) -> dict:
    """Download the exact accepted V4 cache with atomic per-file resume."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_record = _download_file(
        file_name="manifest.json",
        destination=manifest_path,
        expected_sha256=FIXED_500K_MANIFEST_SHA256,
        username=username,
        key=key,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    records = list(manifest["geometry_shards"])
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [
            executor.submit(
                _download_file,
                file_name=record["file"],
                destination=output_root / record["file"],
                expected_sha256=record["sha256"],
                username=username,
                key=key,
            )
            for record in records
        ]
        downloaded = [future.result() for future in futures]
    result = {
        "status": "downloaded",
        "dataset": DATASET_SLUG,
        "root": str(output_root),
        "manifest": manifest_record,
        "files": downloaded,
    }
    atomic_json(output_root / "download_report.json", result)
    return result


def accept_fixed_500k_cache(output_root: Path, *, deep: bool = True) -> dict:
    """Fail closed on file identity, graph counts and source-index ranges."""
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "manifest.json"
    if sha256_file(manifest_path) != FIXED_500K_MANIFEST_SHA256:
        raise RuntimeError("Local fixed-500K manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    checks = []
    for record in manifest["geometry_shards"]:
        path = output_root / record["file"]
        observed = sha256_file(path)
        check = {
            "file": record["file"],
            "bytes": path.stat().st_size,
            "sha256": observed,
            "rows": int(record["rows"]),
            "source_idx_min": int(record["source_idx_min"]),
            "source_idx_max": int(record["source_idx_max"]),
        }
        if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Local fixed-500K shard identity failed: {record['file']}")
        if deep:
            import torch

            data, slices = torch.load(path, map_location="cpu", weights_only=False)
            rows = int(slices["source_idx"].numel() - 1)
            source_idx = data.source_idx.view(-1)
            if rows != record["rows"]:
                raise RuntimeError(f"Local fixed-500K row count failed: {record['file']}")
            if int(source_idx.min()) != record["source_idx_min"] or int(source_idx.max()) != record["source_idx_max"]:
                raise RuntimeError(f"Local fixed-500K source range failed: {record['file']}")
            check["deep_graph_check"] = True
            del data, slices, source_idx
            gc.collect()
        checks.append(check)
    acceptance = {
        "format": "molgap-pcqm-500k-v4-local-acceptance-v1",
        "status": "accepted",
        "manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "train_rows": EXPECTED_ROWS["train"],
        "development_rows": EXPECTED_ROWS["development"],
        "deep_graph_check": bool(deep),
        "shards": checks,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output_root / "local_acceptance.json", acceptance)
    return acceptance


def local_source_sha256(repo_root: Path) -> str:
    """Fingerprint every source file that defines the local V4 ablation run."""
    repo_root = Path(repo_root).resolve()
    files = (
        "src/molgap/pcqm_500k_v4_ablation.py",
        "src/molgap/pcqm_500k_v4_evidence.py",
        "src/molgap/pcqm_k1_scale.py",
        "src/molgap/pcqm_k1_scale_runner.py",
        "src/molgap/qm9_neural_atom.py",
        "src/molgap/pcqm_gap_architecture.py",
        "src/molgap/gps.py",
        "src/molgap/screen_policy.py",
        "src/molgap/training_reproducibility.py",
        "experiments/pcqm_500k_v4_evidence/run_local_ablation.py",
    )
    digest = hashlib.sha256()
    for relative in files:
        path = repo_root / relative
        digest.update(relative.encode("ascii") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()
