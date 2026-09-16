"""Mechanical source/cache/payload gate for the Kunshan Stage-1 audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_k1_explainability import (
    DEVELOPMENT_ROWS,
    EXPECTED_GEOMETRY_SHA256,
    EXPECTED_MANIFEST_SHA256,
    REFERENCE,
    sha256_file,
)
from molgap.pcqm_wedge import WedgeData  # noqa: F401 -- required by cache pickle


REQUIRED_GRAPH_FIELDS = {
    "x",
    "edge_index",
    "edge_attr",
    "random_walk_pe",
    "source_idx",
    "y",
}
REQUIRED_PAYLOAD_FIELDS = {"source_idx", "target_eV", "prediction_eV"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_preflight(cache_root: Path, payload_root: Path, output_path: Path) -> dict:
    manifest_path = cache_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Fixed 100K manifest content changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("geometry_aggregate_sha256") != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError("Fixed geometry aggregate changed")
    for role in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Sealed role flag changed: {role}")

    development = [
        item for item in manifest["geometry_shards"] if item["role"] == "development"
    ]
    if len(development) != 1:
        raise RuntimeError(f"Expected one development shard, found {development}")
    shard = development[0]
    shard_path = cache_root / shard["file"]
    if sha256_file(shard_path) != shard["sha256"]:
        raise RuntimeError("Development shard changed")
    data, slices = torch.load(shard_path, map_location="cpu")
    missing_graph = sorted(REQUIRED_GRAPH_FIELDS.difference(data.keys()))
    if missing_graph:
        raise RuntimeError(f"Development shard fields missing: {missing_graph}")
    if not {"x", "edge_index"}.issubset(slices):
        raise RuntimeError("Development shard slices are incomplete")
    source_idx = data.source_idx.view(-1).long().numpy()
    expected_source = np.arange(100_000, 150_000, dtype=np.int64)
    if len(source_idx) != DEVELOPMENT_ROWS or not np.array_equal(source_idx, expected_source):
        raise RuntimeError("Development source order changed")

    payload_manifest_path = payload_root / "payload_manifest.json"
    payload_manifest = json.loads(payload_manifest_path.read_text(encoding="utf-8"))
    entries = payload_manifest.get("payloads", [])
    if not entries:
        raise RuntimeError("Prediction payload manifest is empty")
    modes = {item["mode"] for item in entries}
    if REFERENCE not in modes:
        raise RuntimeError("Frozen K1-v4 reference payload is absent")
    for item in entries:
        path = payload_root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Prediction payload changed: {item['mode']}")

    reference_entry = next(item for item in entries if item["mode"] == REFERENCE)
    reference_payload = torch.load(
        payload_root / reference_entry["file"], map_location="cpu"
    )
    missing_payload = sorted(REQUIRED_PAYLOAD_FIELDS.difference(reference_payload))
    if missing_payload:
        raise RuntimeError(f"Reference payload fields missing: {missing_payload}")
    for field in REQUIRED_PAYLOAD_FIELDS:
        if reference_payload[field].numel() != DEVELOPMENT_ROWS:
            raise RuntimeError(f"Reference payload row count changed: {field}")

    result = {
        "format": "molgap-pcqm-k1-error-attribution-preflight-v1",
        "accepted": True,
        "model_inference_executed": False,
        "training_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "cache_manifest_sha256": manifest_sha,
        "geometry_aggregate_sha256": manifest["geometry_aggregate_sha256"],
        "development_rows": DEVELOPMENT_ROWS,
        "payload_count": len(entries),
        "payload_manifest_sha256": sha256_file(payload_manifest_path),
        "reference": REFERENCE,
        "deserialized_graph_class": type(data).__name__,
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.cache_root, args.payload_root, args.output)
    print(result["format"], result["accepted"], result["payload_count"])


if __name__ == "__main__":
    main()
