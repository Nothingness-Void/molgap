"""Deterministic data-role bridge from the PCQM 100K screen to 500K.

This module handles row identities only.  It never opens molecular records,
targets, model checkpoints, official validation, or test-dev.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


OFFICIAL_TRAIN_ROWS = 3_378_606
BASE_TRAIN_ROWS = 100_000
SCALE_TRAIN_ROWS = 500_000
VALIDATION_ROWS = 10_000
ADDITIONAL_ROWS = SCALE_TRAIN_ROWS - BASE_TRAIN_ROWS
EXPANSION_SEED = 2_026_091_050
BASE_TRAIN_SHA256 = "d08e04ef73090b77963a6959d7efa22d22a4085e3869a0e13086491b8f8c9678"
VALIDATION_SHA256 = "40b210c03789249f89950d7eb9df6de93ec151903f75077cfa64fe0a94eca5b0"
PARENT_GRAPH_CACHE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
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


def build_scale_split(base_split: dict, shadow_split: dict) -> dict:
    """Append 400K train rows without changing the accepted 10K validation."""
    import numpy as np

    base_train = np.asarray(base_split.get("train", []), dtype=np.int64)
    validation = np.asarray(base_split.get("validation", []), dtype=np.int64)
    base_reserve = np.asarray(base_split.get("reserve", []), dtype=np.int64)
    shadow = np.asarray(shadow_split.get("effective_shadow", []), dtype=np.int64)
    shadow_reserve = np.asarray(shadow_split.get("reserve", []), dtype=np.int64)

    checks = {
        "base_train_rows": base_train.size == BASE_TRAIN_ROWS,
        "validation_rows": validation.size == VALIDATION_ROWS,
        "base_train_sha256": index_sha256(base_train) == BASE_TRAIN_SHA256,
        "validation_sha256": index_sha256(validation) == VALIDATION_SHA256,
        "shadow_rows": shadow.size == 10_000,
        "official_train_boundary": all(
            values.size == 0
            or (int(values.min()) >= 0 and int(values.max()) < OFFICIAL_TRAIN_ROWS)
            for values in (base_train, validation, base_reserve, shadow, shadow_reserve)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Scale split input contract failed: {checks}")

    protected = np.unique(
        np.concatenate((base_train, validation, base_reserve, shadow, shadow_reserve))
    )
    if protected.size != sum(
        values.size
        for values in (base_train, validation, base_reserve, shadow, shadow_reserve)
    ):
        raise RuntimeError("Base, validation, reserve, and shadow roles overlap")
    available = np.setdiff1d(
        np.arange(OFFICIAL_TRAIN_ROWS, dtype=np.int64),
        protected,
        assume_unique=True,
    )
    generator = np.random.default_rng(EXPANSION_SEED)
    added = np.sort(
        generator.choice(available, size=ADDITIONAL_ROWS, replace=False).astype(np.int64)
    )
    train = np.sort(np.concatenate((base_train, added))).astype(np.int64)
    if train.size != SCALE_TRAIN_ROWS or np.unique(train).size != SCALE_TRAIN_ROWS:
        raise RuntimeError("Expanded train role is incomplete or duplicated")
    if np.intersect1d(train, validation).size or np.intersect1d(train, shadow).size:
        raise RuntimeError("Expanded train role overlaps validation or shadow")

    return {
        "format": "molgap-pcqm-k1-scale500k-split-v1",
        "complete": True,
        "official_train_rows": OFFICIAL_TRAIN_ROWS,
        "expansion_seed": EXPANSION_SEED,
        "base_train_rows": BASE_TRAIN_ROWS,
        "added_train_rows": ADDITIONAL_ROWS,
        "train_rows": SCALE_TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "train": train.tolist(),
        "added_train": added.tolist(),
        "validation": validation.tolist(),
        "base_train_sha256": BASE_TRAIN_SHA256,
        "added_train_sha256": index_sha256(added),
        "train_sha256": index_sha256(train),
        "validation_sha256": VALIDATION_SHA256,
        "shadow_sha256": index_sha256(shadow),
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "frozen_k1_source_commit": FROZEN_K1_SOURCE_COMMIT,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
        "molecular_records_read": False,
        "target_labels_read": False,
    }


def write_scale_split(base_split_path: Path, shadow_split_path: Path, output: Path) -> dict:
    base = json.loads(base_split_path.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_split_path.read_text(encoding="utf-8"))
    result = build_scale_split(base, shadow)
    _atomic_json(output, result)
    return result
