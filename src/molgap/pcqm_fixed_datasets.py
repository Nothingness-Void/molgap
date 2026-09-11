"""Immutable views over accepted official PCQM4Mv2 graph caches."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OFFICIAL_ARCHIVE_SHA256 = (
    "628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701"
)
OFFICIAL_ROW_MANIFEST_SHA256 = (
    "c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b"
)
TOPOLOGY_ACCEPTANCE_SHA256 = (
    "83cd367290b0cc5ceb3f49278816d15998bba285afe5286aeacbb6d89e5555ed"
)
GEOMETRY_ACCEPTANCE_SHA256 = (
    "650b1fbd14771888ba1c9342cdcba240b420bd91b0dced974ae654a4b65ab9d2"
)
SCNET_500K_AGGREGATE_SHA256 = (
    "676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20"
)
OFFICIAL_TRAIN_ROWS = 3_378_606
OFFICIAL_VALID_ROWS = 73_545
SHARD_ROWS = 50_000


@dataclass(frozen=True)
class FixedScale:
    name: str
    train_rows: int
    development_rows: int
    kaggle1: bool
    scnet_compatible: bool = False


FIXED_SCALES = (
    FixedScale("ogb-train-100k", 100_000, 50_000, True),
    FixedScale("ogb-train-500k-scnet-v1", 500_000, 50_000, True, True),
    FixedScale("ogb-train-1m", 1_000_000, 50_000, False),
    FixedScale("ogb-train-full", OFFICIAL_TRAIN_ROWS, 0, False),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _inside(path: Path, boundary: Path) -> bool:
    try:
        path.resolve().relative_to(boundary.resolve())
    except ValueError:
        return False
    return True


def _hardlink(source: Path, destination: Path, boundary: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or not _inside(source, boundary):
        raise RuntimeError(f"Source is absent or outside the boundary: {source}")
    if not _inside(destination.parent, boundary):
        raise RuntimeError(f"Destination is outside the boundary: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise RuntimeError(f"Existing fixed asset is not the expected hardlink: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.linking")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def _aggregate(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    return digest.hexdigest()


def _select_train_shards(shards: list[dict], start: int, rows: int) -> list[dict]:
    if start % SHARD_ROWS or (rows % SHARD_ROWS and start + rows != OFFICIAL_TRAIN_ROWS):
        raise ValueError("Fixed ranges must align to accepted 50K shards")
    stop = start + rows
    selected = [
        item
        for item in shards
        if item.get("role") == "train"
        and int(item["source_idx_min"]) >= start
        and int(item["source_idx_max"]) < stop
    ]
    selected.sort(key=lambda item: int(item["source_idx_min"]))
    cursor = start
    observed = 0
    for item in selected:
        if int(item["source_idx_min"]) != cursor:
            raise RuntimeError(f"Accepted train shards are not contiguous at {cursor}")
        cursor = int(item["source_idx_max"]) + 1
        observed += int(item["rows"])
    if cursor != stop or observed != rows:
        raise RuntimeError(
            f"Accepted range {start}:{stop} resolved to {observed} rows ending at {cursor}"
        )
    return selected


def _record(item: dict, *, role: str, modality: str) -> dict:
    if modality == "geometry":
        sha256 = item["sha256"]
        source = item["_geometry_source"]
    elif modality == "topology":
        sha256 = item["base_sha256"]
        source = item["base_path"]
    else:
        raise ValueError(modality)
    return {
        "role": role,
        "file": f"store/{modality}/{item['path']}",
        "rows": int(item["rows"]),
        "source_idx_min": int(item["source_idx_min"]),
        "source_idx_max": int(item["source_idx_max"]),
        "sha256": sha256,
        "bytes": int(Path(source).stat().st_size),
    }


def _file_record(path: Path, *, file: str, sha256: str) -> dict:
    return {
        "file": file,
        "sha256": sha256,
        "bytes": Path(path).stat().st_size,
    }


def build_fixed_pcqm_views(
    *,
    boundary: Path,
    output_root: Path,
    archive: Path,
    row_manifest_path: Path,
    topology_acceptance_path: Path,
    geometry_root: Path,
    geometry_acceptance_path: Path,
) -> dict:
    """Build hardlinked stores and fixed subset manifests without graph recomputation."""
    boundary = Path(boundary).resolve()
    output_root = Path(output_root).resolve()
    inputs = tuple(
        map(
            Path,
            (
                archive,
                row_manifest_path,
                topology_acceptance_path,
                geometry_root,
                geometry_acceptance_path,
            ),
        )
    )
    if not _inside(output_root.parent, boundary) or any(
        not _inside(path, boundary) for path in inputs
    ):
        raise RuntimeError("All fixed-dataset paths must stay inside the approved boundary")

    row_manifest = _load_json(row_manifest_path)
    topology_acceptance = _load_json(topology_acceptance_path)
    geometry_acceptance = _load_json(geometry_acceptance_path)
    checks = {
        "archive_sha256": sha256_file(archive) == OFFICIAL_ARCHIVE_SHA256,
        "row_manifest_sha256": sha256_file(row_manifest_path)
        == OFFICIAL_ROW_MANIFEST_SHA256,
        "topology_acceptance_sha256": sha256_file(topology_acceptance_path)
        == TOPOLOGY_ACCEPTANCE_SHA256,
        "geometry_acceptance_sha256": sha256_file(geometry_acceptance_path)
        == GEOMETRY_ACCEPTANCE_SHA256,
        "row_manifest_complete": row_manifest.get("status") == "complete",
        "topology_accepted": topology_acceptance.get("status") == "accepted",
        "geometry_accepted": geometry_acceptance.get("status") == "accepted",
        "official_archive_identity": row_manifest.get("archive_sha256")
        == OFFICIAL_ARCHIVE_SHA256,
        "official_row_identity": geometry_acceptance.get("row_manifest_sha256")
        == OFFICIAL_ROW_MANIFEST_SHA256
        and topology_acceptance.get("source_manifest_sha256")
        == OFFICIAL_ROW_MANIFEST_SHA256,
        "ogb_features": geometry_acceptance.get("feature_schema") == "ogb"
        and geometry_acceptance.get("node_feature_dim") == 9
        and geometry_acceptance.get("edge_feature_dim") == 3
        and geometry_acceptance.get("rwse_dim") == 16,
        "test_roles_sealed": geometry_acceptance.get("official_test_used") is False
        and topology_acceptance.get("test_graphs_built") is False,
        "no_external_data": geometry_acceptance.get("external_data_used") is False
        and topology_acceptance.get("external_data_used") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Official PCQM source acceptance failed: {checks}")

    geometry_shards = []
    for raw in geometry_acceptance["shards"]:
        item = dict(raw)
        item["_geometry_source"] = str(Path(geometry_root) / item["path"])
        geometry_shards.append(item)

    # Materialize one content store. Subsets below are manifests over this store.
    _hardlink(archive, output_root / "store/source/pcqm4m-v2.zip", boundary)
    _hardlink(
        row_manifest_path, output_root / "store/source/rows/manifest.json", boundary
    )
    _hardlink(
        topology_acceptance_path,
        output_root / "store/topology/acceptance.json",
        boundary,
    )
    _hardlink(
        geometry_acceptance_path,
        output_root / "store/geometry/acceptance.json",
        boundary,
    )
    row_root = row_manifest_path.parent
    for item in row_manifest["shards"]:
        _hardlink(
            row_root / item["path"],
            output_root / "store/source/rows" / item["path"],
            boundary,
        )
    for item in geometry_shards:
        _hardlink(
            Path(item["base_path"]),
            output_root / "store/topology" / item["path"],
            boundary,
        )
        _hardlink(
            Path(item["_geometry_source"]),
            output_root / "store/geometry" / item["path"],
            boundary,
        )

    store_assets = {
        "source": [
            _file_record(
                archive,
                file="store/source/pcqm4m-v2.zip",
                sha256=OFFICIAL_ARCHIVE_SHA256,
            ),
            _file_record(
                row_manifest_path,
                file="store/source/rows/manifest.json",
                sha256=OFFICIAL_ROW_MANIFEST_SHA256,
            ),
            *[
                _file_record(
                    row_root / item["path"],
                    file=f"store/source/rows/{item['path']}",
                    sha256=item["sha256"],
                )
                for item in row_manifest["shards"]
            ],
        ],
        "topology": [
            _file_record(
                topology_acceptance_path,
                file="store/topology/acceptance.json",
                sha256=TOPOLOGY_ACCEPTANCE_SHA256,
            ),
            *[
                _record(item, role=item["role"], modality="topology")
                for item in geometry_shards
            ],
        ],
        "geometry": [
            _file_record(
                geometry_acceptance_path,
                file="store/geometry/acceptance.json",
                sha256=GEOMETRY_ACCEPTANCE_SHA256,
            ),
            *[
                _record(item, role=item["role"], modality="geometry")
                for item in geometry_shards
            ],
        ],
    }

    subset_summaries = []
    for spec in FIXED_SCALES:
        train = _select_train_shards(geometry_shards, 0, spec.train_rows)
        development = (
            _select_train_shards(
                geometry_shards, spec.train_rows, spec.development_rows
            )
            if spec.development_rows
            else []
        )
        assets = {"topology": [], "geometry": []}
        for modality in assets:
            assets[modality].extend(
                _record(item, role="train", modality=modality) for item in train
            )
            assets[modality].extend(
                _record(item, role="development", modality=modality)
                for item in development
            )
        payload = {
            "format": "molgap-pcqm4mv2-fixed-subset-v1",
            "status": "complete",
            "identity": asdict(spec),
            "source": {
                "dataset": "OGB-LSC PCQM4Mv2",
                "official_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
                "official_row_manifest_sha256": OFFICIAL_ROW_MANIFEST_SHA256,
                "official_train_rows": OFFICIAL_TRAIN_ROWS,
                "official_valid_rows": OFFICIAL_VALID_ROWS,
                "external_data_used": False,
            },
            "roles": {
                "train": {
                    "source_idx_start": 0,
                    "source_idx_stop": spec.train_rows,
                    "rows": spec.train_rows,
                },
                "development": (
                    {
                        "source_idx_start": spec.train_rows,
                        "source_idx_stop": spec.train_rows + spec.development_rows,
                        "rows": spec.development_rows,
                    }
                    if spec.development_rows
                    else None
                ),
            },
            "graph_contract": {
                "feature_schema": "ogb",
                "node_feature_dim": 9,
                "edge_feature_dim": 3,
                "rwse_dim": 16,
                "geometry_method": "ETKDGv3",
                "optimization_method": "MMFF94s",
            },
            "assets": assets,
            "aggregates": {
                modality: _aggregate(records) for modality, records in assets.items()
            },
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }
        if spec.scnet_compatible:
            scnet_records = [
                {
                    **item,
                    "file": item["file"].removeprefix("store/geometry/"),
                }
                for item in assets["geometry"]
            ]
            scnet_aggregate = _aggregate(scnet_records)
            if scnet_aggregate != SCNET_500K_AGGREGATE_SHA256:
                raise RuntimeError(f"SCNet 500K identity changed: {scnet_aggregate}")
            payload["scnet_compatibility"] = {
                "format": "molgap-pcqm-edgestate304-500k-subset-v1",
                "aggregate_sha256": scnet_aggregate,
                "bytewise_identical_geometry_shards": True,
            }
        manifest_path = output_root / "subsets" / spec.name / "manifest.json"
        atomic_json(manifest_path, payload)
        subset_summaries.append(
            {
                "name": spec.name,
                "manifest": str(manifest_path.relative_to(output_root)),
                "manifest_sha256": sha256_file(manifest_path),
                "train_rows": spec.train_rows,
                "development_rows": spec.development_rows,
                "kaggle1": spec.kaggle1,
            }
        )

    root_manifest = {
        "format": "molgap-pcqm4mv2-fixed-dataset-root-v1",
        "status": "complete",
        "materialization": "hardlink-content-store-plus-manifest-views",
        "checks": checks,
        "store_assets": store_assets,
        "subsets": subset_summaries,
        "kaggle1_policy": {
            "allowed_subsets": ["ogb-train-100k", "ogb-train-500k-scnet-v1"],
            "excluded_subsets": ["ogb-train-1m", "ogb-train-full"],
        },
    }
    atomic_json(output_root / "manifest.json", root_manifest)
    return root_manifest


def accept_fixed_pcqm_views(output_root: Path, *, verify_content: bool) -> dict:
    output_root = Path(output_root).resolve()
    root = _load_json(output_root / "manifest.json")
    checks = {
        "root_complete": root.get("status") == "complete",
        "four_subsets": len(root.get("subsets", [])) == 4,
        "kaggle_policy": root.get("kaggle1_policy", {}).get("allowed_subsets")
        == ["ogb-train-100k", "ogb-train-500k-scnet-v1"],
    }
    files_checked = 0
    bytes_checked = 0
    seen: set[str] = set()
    for category, records in root.get("store_assets", {}).items():
        for item in records:
            if item["file"] in seen:
                continue
            seen.add(item["file"])
            path = output_root / item["file"]
            ok = path.is_file() and path.stat().st_size == int(item["bytes"])
            if ok and verify_content:
                ok = sha256_file(path) == item["sha256"]
                bytes_checked += int(item["bytes"])
            checks[f"store_{category}_{files_checked:04d}"] = ok
            files_checked += 1
    for summary in root.get("subsets", []):
        manifest_path = output_root / summary["manifest"]
        manifest = _load_json(manifest_path)
        checks[f"manifest_{summary['name']}"] = (
            sha256_file(manifest_path) == summary["manifest_sha256"]
            and manifest.get("status") == "complete"
        )
        for modality, records in manifest.get("assets", {}).items():
            checks[f"aggregate_{summary['name']}_{modality}"] = (
                _aggregate(records) == manifest["aggregates"][modality]
            )
            for item in records:
                if item["file"] in seen:
                    continue
                seen.add(item["file"])
                path = output_root / item["file"]
                ok = path.is_file() and path.stat().st_size == int(item["bytes"])
                if ok and verify_content:
                    ok = sha256_file(path) == item["sha256"]
                    bytes_checked += int(item["bytes"])
                checks[f"file_{files_checked:04d}"] = ok
                files_checked += 1
    acceptance = {
        "format": "molgap-pcqm4mv2-fixed-dataset-acceptance-v1",
        "status": "accepted" if all(checks.values()) else "rejected",
        "verify_content": verify_content,
        "files_checked": files_checked,
        "bytes_checked": bytes_checked,
        "checks": checks,
    }
    atomic_json(output_root / "acceptance.json", acceptance)
    if acceptance["status"] != "accepted":
        raise RuntimeError("Fixed PCQM dataset acceptance failed")
    return acceptance
