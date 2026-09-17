from __future__ import annotations

import json
from pathlib import Path

import pytest

from molgap import kaggle_fixed_mirror


def _payload(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "payload"
    shard = root / "train" / "train_shard_0000.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"graph-bytes")
    manifest = {
        "format": "molgap-pcqm4mv2-kaggle-fixed-subset-v1",
        "status": "complete",
        "identity": {
            "name": "fixture",
            "train_rows": 1,
            "development_rows": 0,
        },
        "geometry_shards": [
            {
                "file": "train/train_shard_0000.pt",
                "rows": 1,
                "bytes": shard.stat().st_size,
                "sha256": kaggle_fixed_mirror.sha256_file(shard),
            }
        ],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_prepare_account_mirror_verifies_and_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _payload(tmp_path)
    monkeypatch.setitem(
        kaggle_fixed_mirror.FIXED_MIRRORS,
        "fixture",
        {
            "slug": "fixed-fixture",
            "title": "Fixed Fixture",
            "manifest_sha256": kaggle_fixed_mirror.sha256_file(
                root / "manifest.json"
            ),
            "identity_name": "fixture",
            "train_rows": 1,
            "development_rows": 0,
            "graph_shards": 1,
        },
    )

    report = kaggle_fixed_mirror.prepare_account_mirror(
        root, "account-name", "fixture"
    )

    assert report["status"] == "accepted_for_mirror"
    assert report["protected_roles_included"] is False
    metadata = json.loads((root / "dataset-metadata.json").read_text())
    assert metadata["id"] == "account-name/fixed-fixture"
    assert metadata["isPrivate"] is True
    assert manifest["official_validation_role_read"] is False


def test_prepare_account_mirror_rejects_protected_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _payload(tmp_path)
    manifest["official_validation_role_read"] = True
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setitem(
        kaggle_fixed_mirror.FIXED_MIRRORS,
        "fixture",
        {
            "slug": "fixed-fixture",
            "title": "Fixed Fixture",
            "manifest_sha256": kaggle_fixed_mirror.sha256_file(
                root / "manifest.json"
            ),
            "identity_name": "fixture",
            "train_rows": 1,
            "development_rows": 0,
            "graph_shards": 1,
        },
    )

    with pytest.raises(RuntimeError, match="protected"):
        kaggle_fixed_mirror.verify_fixed_mirror(root, "fixture")


def test_verify_published_mirror_accepts_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _payload(tmp_path)
    manifest_sha256 = kaggle_fixed_mirror.sha256_file(root / "manifest.json")
    monkeypatch.setitem(
        kaggle_fixed_mirror.FIXED_MIRRORS,
        "fixture",
        {
            "slug": "fixed-fixture",
            "title": "Fixed Fixture",
            "manifest_sha256": manifest_sha256,
            "identity_name": "fixture",
            "train_rows": 1,
            "development_rows": 0,
            "graph_shards": 1,
        },
    )
    metadata = {
        "datasetId": 123,
        "datasetSlug": "fixed-fixture",
        "ownerUser": "account-name",
        "isPrivate": True,
    }
    (root / "dataset-metadata.json").write_text(
        json.dumps(json.dumps(metadata)), encoding="utf-8"
    )
    (root / "README.md").write_bytes(b"x" * 490)
    shard = root / "train" / "train_shard_0000.pt"
    (root / "remote_files.csv").write_text(
        "name,size,creationDate\n"
        "README.md,490,now\n"
        f"manifest.json,{(root / 'manifest.json').stat().st_size},now\n"
        f"train/train_shard_0000.pt,{shard.stat().st_size},now\n",
        encoding="utf-8",
    )

    result = kaggle_fixed_mirror.verify_published_mirror(
        manifest_path=root / "manifest.json",
        remote_files_path=root / "remote_files.csv",
        remote_metadata_path=root / "dataset-metadata.json",
        account="account-name",
        role="fixture",
    )

    assert result["status"] == "accepted"
    assert result["dataset_id"] == 123
    assert result["remote_file_inventory_matches"] is True
