"""Lightweight QM9 source acquisition and immutable split primitives."""
from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


QM9_PROCESSED_URL = "https://data.pyg.org/datasets/qm9_v3.zip"
QM9_RAW_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/molnet_publish/qm9.zip"
)
DEFAULT_CACHE = Path("data/cache/qm9")


@dataclass(frozen=True)
class ScreenSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int

    @property
    def all_indices(self) -> np.ndarray:
        return np.concatenate((self.train, self.validation, self.test))

    @property
    def fingerprint(self) -> str:
        value = self.all_indices.astype(np.int64).tobytes()
        return hashlib.sha256(value).hexdigest()[:16]


def fixed_split(
    n_total: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> ScreenSplit:
    requested = train_size + validation_size + test_size
    if requested > n_total:
        raise ValueError(f"Requested {requested} rows from QM9 with {n_total} rows")
    order = np.random.RandomState(seed).permutation(n_total)[:requested]
    train_end = train_size
    validation_end = train_end + validation_size
    return ScreenSplit(
        train=order[:train_end],
        validation=order[train_end:validation_end],
        test=order[validation_end:],
        seed=seed,
    )


def fixed_split_from_pool(
    source_indices: np.ndarray,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> ScreenSplit:
    """Select deterministic source indices from an explicitly accepted pool."""
    pool = np.asarray(source_indices, dtype=np.int64)
    if pool.ndim != 1 or len(np.unique(pool)) != len(pool):
        raise ValueError("QM9 source pool must be one-dimensional and unique")
    positions = fixed_split(
        len(pool), train_size, validation_size, test_size, seed
    )
    return ScreenSplit(
        train=pool[positions.train],
        validation=pool[positions.validation],
        test=pool[positions.test],
        seed=seed,
    )


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def prepare_qm9_processed(cache_dir: Path = DEFAULT_CACHE) -> Path:
    processed = cache_dir / "preprocessed" / "qm9_v3.pt"
    if not processed.exists():
        archive = cache_dir / "download" / "qm9_v3.zip"
        _download(QM9_PROCESSED_URL, archive)
        processed.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(processed.parent)
    return processed


def prepare_qm9_files(cache_dir: Path = DEFAULT_CACHE) -> dict[str, Path]:
    processed = prepare_qm9_processed(cache_dir)
    raw_sdf = cache_dir / "raw" / "gdb9.sdf"
    if not raw_sdf.exists():
        archive = cache_dir / "download" / "qm9_raw.zip"
        _download(QM9_RAW_URL, archive)
        raw_sdf.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(raw_sdf.parent)
    return {"processed": processed, "raw_sdf": raw_sdf}


def load_qm9_records(cache_dir: Path = DEFAULT_CACHE) -> list[dict]:
    processed = prepare_qm9_processed(cache_dir)
    records = torch.load(processed, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Unexpected QM9 payload: {processed}")
    return records
