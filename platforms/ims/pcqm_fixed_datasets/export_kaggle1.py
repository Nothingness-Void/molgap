from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from molgap.pcqm_fixed_datasets import atomic_json, sha256_file


KAGGLE_IDENTITIES = {
    "ogb-train-100k": {
        "id": "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1",
        "title": "PCQM4Mv2 OGB Fixed 100K V1",
    },
    "ogb-train-500k-scnet-v1": {
        "id": "nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "title": "PCQM4Mv2 OGB Fixed 500K SCNet V1",
    },
}


def _inside(path: Path, boundary: Path) -> bool:
    try:
        path.resolve().relative_to(boundary.resolve())
    except ValueError:
        return False
    return True


def _link(source: Path, destination: Path, boundary: Path) -> None:
    if not source.is_file() or not _inside(source, boundary):
        raise RuntimeError(f"Invalid source: {source}")
    if not _inside(destination.parent, boundary):
        raise RuntimeError(f"Invalid destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise RuntimeError(f"Refusing to overwrite {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.linking")
    os.link(source, temporary)
    os.replace(temporary, destination)


def export_one(fixed_root: Path, export_root: Path, name: str) -> dict:
    acceptance = json.loads(
        (fixed_root / "acceptance.json").read_text(encoding="utf-8")
    )
    if acceptance.get("status") != "accepted" or not acceptance.get("verify_content"):
        raise RuntimeError("Fixed root lacks full content acceptance")
    subset_path = fixed_root / "subsets" / name / "manifest.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if not subset.get("identity", {}).get("kaggle1"):
        raise RuntimeError(f"Subset is excluded from Kaggle1: {name}")
    destination = export_root / name
    destination.mkdir(parents=True, exist_ok=True)
    shards = []
    for item in subset["assets"]["geometry"]:
        relative = item["file"].removeprefix("store/geometry/")
        _link(fixed_root / item["file"], destination / relative, fixed_root.parent)
        shards.append({**item, "file": relative})
    manifest = {
        "format": "molgap-pcqm4mv2-kaggle-fixed-subset-v1",
        "status": "complete",
        "identity": subset["identity"],
        "source": subset["source"],
        "roles": subset["roles"],
        "graph_contract": subset["graph_contract"],
        "geometry_shards": shards,
        "source_rows_included": False,
        "source_indices_embedded_in_graphs": True,
        "geometry_aggregate_sha256": subset["aggregates"]["geometry"],
        "scnet_compatibility": subset.get("scnet_compatibility"),
        "fixed_root_manifest_sha256": sha256_file(fixed_root / "manifest.json"),
        "fixed_root_acceptance_sha256": sha256_file(
            fixed_root / "acceptance.json"
        ),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(destination / "manifest.json", manifest)
    metadata = {
        **KAGGLE_IDENTITIES[name],
        "licenses": [{"name": "CC-BY-4.0"}],
        "isPrivate": True,
    }
    atomic_json(destination / "dataset-metadata.json", metadata)
    (destination / "README.md").write_text(
        "# Fixed OGB PCQM4Mv2 cache\n\n"
        "Derived only from the official OGB-LSC PCQM4Mv2 training role. "
        "The graph cache uses OGB categorical atom/bond features, RWSE16, "
        "and ETKDGv3+MMFF94s geometry. See `manifest.json` for exact row "
        "ranges and SHA256 identities. Source indices and labels are embedded "
        "in every graph; raw row shards remain in the IMS canonical store. "
        "Official validation and test roles are not included. PCQM4Mv2 is "
        "licensed CC BY 4.0; cite OGB-LSC "
        "and PubChemQC when using this cache.\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "directory": str(destination),
        "files": len(shards) + 3,
        "payload_bytes": sum(int(item["bytes"]) for item in shards),
        "manifest_sha256": sha256_file(destination / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    results = [
        export_one(args.fixed_root.resolve(), args.export_root.resolve(), name)
        for name in KAGGLE_IDENTITIES
    ]
    atomic_json(args.export_root / "manifest.json", {"status": "complete", "exports": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
