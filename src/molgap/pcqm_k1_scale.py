"""Exact row contract shared by SCNet and the Kaggle K1 500K benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OFFICIAL_TRAIN_ROWS = 3_378_606
SCALE_TRAIN_ROWS = 500_000
VALIDATION_ROWS = 50_000
ROLE_ROWS_READ = SCALE_TRAIN_ROWS + VALIDATION_ROWS
TRAIN_SHA256 = "a9c8b2b698c67f30348c6edbccff00eb9e2c06b064ee4d1a11e607f9531a0f8e"
VALIDATION_SHA256 = "9ae885e5e74d82820d83758942eaf3e6ae2a7dcefbc1f0f4c174ce94c6786bb9"
SCNET_REFERENCE_CACHE_SHA256 = (
    "676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20"
)
UNSANITIZED_OGB_SOURCE_INDICES = (
    51_128,
    142_541,
    155_174,
    155_175,
    155_176,
    155_180,
    155_181,
    193_600,
    411_357,
    505_556,
)
UNSANITIZED_OGB_INDEX_SHA256 = (
    "93e7e2478d86c41c598c833225af5fcbd914f3e4ba3cced87a7fe9f469108097"
)
FROZEN_K1_SOURCE_COMMIT = "47f99cf9da7fee306f5165175b4020c6c4aa9fb3"


def index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_scale_split() -> dict:
    """Return the exact accepted SCNet 500K/50K source-index roles."""
    train = list(range(SCALE_TRAIN_ROWS))
    validation = list(range(SCALE_TRAIN_ROWS, ROLE_ROWS_READ))
    if index_sha256(train) != TRAIN_SHA256:
        raise RuntimeError("K1 scale train identity changed")
    if index_sha256(validation) != VALIDATION_SHA256:
        raise RuntimeError("K1 scale development identity changed")
    return {
        "format": "molgap-pcqm-k1-scale500k-split-v2",
        "complete": True,
        "official_train_rows": OFFICIAL_TRAIN_ROWS,
        "role_rows_read": ROLE_ROWS_READ,
        "train_rows": SCALE_TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "train": train,
        "validation": validation,
        "train_sha256": TRAIN_SHA256,
        "validation_sha256": VALIDATION_SHA256,
        "selection": "official-train-prefix-500k-next-50k-development",
        "scnet_reference_cache_aggregate_sha256": SCNET_REFERENCE_CACHE_SHA256,
        "frozen_k1_source_commit": FROZEN_K1_SOURCE_COMMIT,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_records_read": False,
        "target_labels_read": False,
    }


def write_scale_split(output: Path) -> dict:
    result = build_scale_split()
    _atomic_json(output, result)
    return result
