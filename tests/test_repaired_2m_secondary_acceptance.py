from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from molgap.artifact_acceptance import accept_repaired_3d_secondary_graphs

SEED = 314159


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_csv(path: Path, rows: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cid", "canonical_smiles", "homo", "lumo", "gap"]
        )
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "cid": 1000 + index,
                    "canonical_smiles": f"C{'C' * (index % 3)}",
                    "homo": -6.0 - index * 0.01,
                    "lumo": -1.0 + index * 0.01,
                    "gap": 5.0 + index * 0.02,
                }
            )


def graph(index: int, offset: float) -> Data:
    return Data(
        z=torch.tensor([6, 6]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0 + offset, 0.0, 0.0]]),
        y=torch.tensor([[-6.0 - index * 0.01, -1.0 + index * 0.01, 5.0 + index * 0.02]]),
        source_idx=torch.tensor([index]),
    )


def build_view(
    root: Path,
    indices: list[int],
    offset: float,
    *,
    view: str,
    primary_hashes: dict[str, str] | None = None,
    failures: list[int] | None = None,
) -> Path:
    shard_dir = root / "graph_shards"
    (shard_dir / "reports").mkdir(parents=True, exist_ok=True)
    name = "graphs_0000000_0000010.pt"
    path = shard_dir / name
    torch.save([graph(i, offset) for i in indices], path)
    report = {
        "status": "complete",
        "requested": len(indices) + len(failures or []),
        "reused": 0,
        "built": len(indices),
        "failed": len(failures or []),
        "failure_source_idx": failures or [],
        "graphs": len(indices),
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if view == "secondary":
        report["view"] = "independent_secondary_etkdg"
        report["seed"] = SEED
        report["primary_shard_sha256"] = (primary_hashes or {})[name]
    (shard_dir / "reports" / f"{path.stem}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return shard_dir


def write_primary_manifest(root: Path, shard_dir: Path, rows: int) -> Path:
    name = "graphs_0000000_0000010.pt"
    manifest = {
        "accepted": True,
        "shards": [{"path": name, "rows": rows, "sha256": sha256(shard_dir / name)}],
        "accepted_rows": rows,
    }
    path = root / "primary_acceptance.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def views(tmp_path: Path):
    source_csv = tmp_path / "source.csv"
    write_source_csv(source_csv, 10)
    kept = [0, 1, 2, 3, 4, 5, 6, 7]
    primary_root = tmp_path / "primary"
    primary_shards = build_view(primary_root, kept, 0.0, view="primary")
    primary_manifest = write_primary_manifest(primary_root, primary_shards, len(kept))
    hashes = {"graphs_0000000_0000010.pt": sha256(primary_shards / "graphs_0000000_0000010.pt")}
    return tmp_path, source_csv, kept, primary_manifest, primary_shards, hashes


def test_accepts_an_independent_secondary_view(views) -> None:
    tmp_path, source_csv, kept, primary_manifest, primary_shards, hashes = views
    secondary = build_view(
        tmp_path / "secondary", kept, 0.5, view="secondary", primary_hashes=hashes
    )
    manifest = accept_repaired_3d_secondary_graphs(
        secondary,
        source_csv,
        primary_manifest,
        tmp_path / "secondary_acceptance.json",
        expected_shards=1,
        primary_shard_dir=primary_shards,
    )
    assert manifest["accepted"] is True
    assert manifest["accepted_rows"] == len(kept)
    assert manifest["requested_rows"] == manifest["primary_accepted_rows"]
    assert manifest["coordinates_compared"] == len(kept)
    assert manifest["coordinates_distinct"] == len(kept)
    assert manifest["coordinates_distinct_fraction"] == 1.0
    assert (tmp_path / "secondary_acceptance.json").is_file()


def test_rejects_a_copied_primary_view(views) -> None:
    # A byte-identical rebuild would satisfy every count and hash check, so the
    # coordinate-independence gate is the only thing standing between a copied
    # cache and a second conformer view.
    tmp_path, source_csv, kept, primary_manifest, primary_shards, hashes = views
    secondary = build_view(
        tmp_path / "secondary", kept, 0.0, view="secondary", primary_hashes=hashes
    )
    with pytest.raises(ValueError, match="not independent"):
        accept_repaired_3d_secondary_graphs(
            secondary,
            source_csv,
            primary_manifest,
            tmp_path / "out.json",
            expected_shards=1,
            primary_shard_dir=primary_shards,
        )


def test_rejects_a_shard_bound_to_an_unaccepted_primary_hash(views) -> None:
    tmp_path, source_csv, kept, primary_manifest, primary_shards, _ = views
    secondary = build_view(
        tmp_path / "secondary",
        kept,
        0.5,
        view="secondary",
        primary_hashes={"graphs_0000000_0000010.pt": "0" * 64},
    )
    with pytest.raises(ValueError, match="unaccepted primary hash"):
        accept_repaired_3d_secondary_graphs(
            secondary,
            source_csv,
            primary_manifest,
            tmp_path / "out.json",
            expected_shards=1,
            primary_shard_dir=primary_shards,
        )


def test_rejects_a_molecule_set_that_disagrees_with_the_primary(views) -> None:
    tmp_path, source_csv, kept, primary_manifest, primary_shards, hashes = views
    secondary = build_view(
        tmp_path / "secondary", kept[:-1], 0.5, view="secondary", primary_hashes=hashes
    )
    with pytest.raises(ValueError, match="requested count does not match"):
        accept_repaired_3d_secondary_graphs(
            secondary,
            source_csv,
            primary_manifest,
            tmp_path / "out.json",
            expected_shards=1,
            primary_shard_dir=primary_shards,
        )


def test_rejects_reused_coordinates(views) -> None:
    tmp_path, source_csv, kept, primary_manifest, primary_shards, hashes = views
    secondary = build_view(
        tmp_path / "secondary", kept, 0.5, view="secondary", primary_hashes=hashes
    )
    sidecar = secondary / "reports" / "graphs_0000000_0000010.json"
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    report["reused"] = 1
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="reused coordinates"):
        accept_repaired_3d_secondary_graphs(
            secondary,
            source_csv,
            primary_manifest,
            tmp_path / "out.json",
            expected_shards=1,
            primary_shard_dir=primary_shards,
        )
