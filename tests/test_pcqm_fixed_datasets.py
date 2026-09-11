from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from molgap import pcqm_fixed_datasets as fixed


EXPORTER = (
    Path(__file__).resolve().parents[1]
    / "platforms"
    / "ims"
    / "pcqm_fixed_datasets"
    / "export_kaggle1.py"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scale_contract_is_nested_and_kaggle_is_bounded() -> None:
    assert [(item.train_rows, item.development_rows) for item in fixed.FIXED_SCALES] == [
        (100_000, 50_000),
        (500_000, 50_000),
        (1_000_000, 50_000),
        (3_378_606, 0),
    ]
    assert [item.name for item in fixed.FIXED_SCALES if item.kaggle1] == [
        "ogb-train-100k",
        "ogb-train-500k-scnet-v1",
    ]
    source = EXPORTER.read_text(encoding="utf-8")
    assert '"ogb-train-100k"' in source
    assert '"ogb-train-500k-scnet-v1"' in source
    assert '"ogb-train-1m"' not in source
    assert '"ogb-train-full"' not in source
    assert '"CC-BY-4.0"' in source
    assert '"isPrivate": True' in source
    assert '"source_rows_included": False' in source
    assert "row_shards" not in source


def test_aggregate_matches_scnet_contract() -> None:
    records = []
    for index in range(11):
        records.append(
            {
                "role": "train" if index < 10 else "development",
                "file": f"train/train_shard_{index:04d}.pt",
                "sha256": str(index),
            }
        )
    expected = hashlib.sha256()
    for item in records:
        expected.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    assert fixed._aggregate(records) == expected.hexdigest()


def test_hardlink_is_create_only_and_boundary_checked(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")
    destination = tmp_path / "fixed" / "asset.bin"
    fixed._hardlink(source, destination, tmp_path)
    assert os.path.samefile(source, destination)
    fixed._hardlink(source, destination, tmp_path)

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"different")
    destination.unlink()
    destination.write_bytes(b"occupied")
    with pytest.raises(RuntimeError, match="not the expected hardlink"):
        fixed._hardlink(replacement, destination, tmp_path)


def test_acceptance_rejects_changed_manifest(tmp_path: Path) -> None:
    subset = tmp_path / "subsets" / "ogb-train-100k"
    subset.mkdir(parents=True)
    payload = {
        "status": "complete",
        "assets": {"topology": [], "geometry": []},
        "aggregates": {
            "topology": hashlib.sha256().hexdigest(),
            "geometry": hashlib.sha256().hexdigest(),
        },
    }
    manifest = subset / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    root = {
        "status": "complete",
        "subsets": [
            {
                "name": "ogb-train-100k",
                "manifest": "subsets/ogb-train-100k/manifest.json",
                "manifest_sha256": _sha(manifest),
            }
        ],
        "kaggle1_policy": {
            "allowed_subsets": ["ogb-train-100k", "ogb-train-500k-scnet-v1"]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(RuntimeError, match="acceptance failed"):
        fixed.accept_fixed_pcqm_views(tmp_path, verify_content=False)
