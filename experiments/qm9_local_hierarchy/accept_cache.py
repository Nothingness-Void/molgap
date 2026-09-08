"""Independent data-only acceptance for the QM9 EdgeState hierarchy cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


EXPECTED_SOURCE_ROWS = 130_831
EXPECTED_RDKIT_VERSION = "2023.09.6"
EXPECTED_PROCESSED_SHA256 = (
    "90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721"
)
EXPECTED_RAW_SDF_SHA256 = (
    "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indices_sha256(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def accept(root: Path, *, source_commit: str) -> dict:
    manifest_path = root / "manifest.json"
    validity_path = root / "canonical_validity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-edgestate-local-hierarchy-cache-v2",
        "complete": True,
        "source_commit": source_commit,
        "roles": {"train": 30_000, "validation": 3_000},
        "held_out_indices_materialized": False,
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": 16,
        "gpu_used": False,
        "model_inference_executed": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Cache contract changed for {key}")
    if sha256_file(validity_path) != manifest["canonical_validity_sha256"]:
        raise RuntimeError("Canonical-validity hash changed")
    if validity.get("format") != "molgap-qm9-canonical-valid-pool-v1":
        raise RuntimeError("Canonical-validity format changed")
    valid = validity["valid_source_indices"]
    train = validity["selected_train_indices"]
    validation = validity["selected_validation_indices"]
    checks = {
        "source_rows": validity["total_source_rows"] == EXPECTED_SOURCE_ROWS,
        "rdkit_version": validity["rdkit_version"] == EXPECTED_RDKIT_VERSION,
        "processed_source": validity["processed_source_sha256"]
        == EXPECTED_PROCESSED_SHA256,
        "raw_sdf_source": validity["raw_sdf_sha256"]
        == EXPECTED_RAW_SDF_SHA256,
        "valid_pool_count": len(valid) == manifest["canonical_valid_pool_count"],
        "valid_pool_hash": indices_sha256(valid)
        == manifest["canonical_valid_pool_sha256"],
        "invalid_count": len(validity["invalid_records"])
        == manifest["canonical_invalid_count"],
        "train_hash": indices_sha256(train)
        == manifest["train_source_indices_sha256"],
        "validation_hash": indices_sha256(validation)
        == manifest["validation_source_indices_sha256"],
        "roles_disjoint": not set(train).intersection(validation),
        "selected_from_pool": set(train).union(validation).issubset(set(valid)),
        "held_out_graphs_absent": validity["held_out_graphs_materialized"] is False,
        "test_role_unread": validity["test_role_read"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Canonical-validity checks failed: {checks}")

    expected_by_role = {"train": train, "validation": validation}
    observed_by_role = {"train": [], "validation": []}
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        observed_sha = sha256_file(path)
        if observed_sha != shard["sha256"]:
            raise RuntimeError(f"Shard hash changed: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload) != shard["graph_count"]:
            raise RuntimeError(f"Shard count changed: {path.name}")
        role = shard["role"]
        observed_by_role[role].extend(
            int(graph.row_id.view(-1)[0]) for graph in payload
        )
        aggregate.update(
            f"{role}\t{shard['file']}\t{observed_sha}\n".encode("ascii")
        )
    for role, expected in expected_by_role.items():
        if observed_by_role[role] != expected:
            raise RuntimeError(f"Source identity changed for {role}")
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Aggregate cache hash changed")

    return {
        "format": "molgap-qm9-edgestate-local-hierarchy-cache-acceptance-v2",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "canonical_validity_sha256": manifest["canonical_validity_sha256"],
        "canonical_valid_pool_count": manifest["canonical_valid_pool_count"],
        "canonical_invalid_count": manifest["canonical_invalid_count"],
        "roles": manifest["roles"],
        "checks": checks,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.root, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
