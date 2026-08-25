"""CPU-side join of accepted Route B 2D and ETKDG primary graph views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from molgap.pcqm_expert import PackedGraphDataset, _save_packed_graphs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = args.source_root / "manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise RuntimeError("The accepted Route B graph cache is not complete")
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_manifest_sha256") != sha256_file(source_manifest):
            raise RuntimeError("TGT-hybrid cache source manifest changed")
    else:
        manifest = {
            "format": "molgap-pcqm-tgt-hybrid-v3-joined-cache-v1",
            "status": "building",
            "source_root": str(args.source_root),
            "source_manifest_sha256": sha256_file(source_manifest),
            "view_contract": {
                "topology": "GPS 2D x/edge/edge_attr plus explicit counts",
                "geometry": "primary ETKDG z/pos plus explicit counts",
            },
            "files": [],
        }
    existing = {item["path"]: item for item in manifest["files"]}
    for role in ("train", "dev", "official"):
        gps_paths = sorted((args.source_root / "gps").glob(f"{role}_shard_*.pt"))
        primary_paths = sorted((args.source_root / "primary").glob(f"{role}_shard_*.pt"))
        if [path.name for path in gps_paths] != [path.name for path in primary_paths]:
            raise RuntimeError(f"GPS/primary shard names differ for {role}")
        for gps_path, primary_path in zip(gps_paths, primary_paths):
            output_path = args.output_root / role / primary_path.name
            relative = output_path.relative_to(args.output_root).as_posix()
            if relative in existing and output_path.exists():
                if sha256_file(output_path) == existing[relative]["sha256"]:
                    continue
                raise RuntimeError(f"Joined shard hash changed: {output_path}")
            gps = PackedGraphDataset(gps_path)
            primary = PackedGraphDataset(primary_path)
            gps_by_source = {
                int(gps[index].source_idx.view(-1)[0]): gps[index]
                for index in range(len(gps))
            }
            joined = []
            for index in range(len(primary)):
                graph = primary[index].clone()
                source_idx = int(graph.source_idx.view(-1)[0])
                topology = gps_by_source.get(source_idx)
                if topology is None:
                    raise RuntimeError(f"Missing aligned GPS row {source_idx}")
                if topology.x.shape[-1] != 18:
                    raise RuntimeError("Expected the accepted 18-wide PCQM node schema")
                # GPS 2D nodes and primary ETKDG explicit-H coordinates are
                # different views; retain both with explicit batch metadata.
                graph.topology_x = topology.x.float().contiguous()
                graph.topology_edges = topology.edge_index.t().contiguous()
                graph.topology_edge_attr = topology.edge_attr.float().contiguous()
                graph.topology_node_count = torch.tensor(
                    [topology.x.shape[0]], dtype=torch.long
                )
                graph.topology_edge_count = torch.tensor(
                    [topology.edge_index.shape[1]], dtype=torch.long
                )
                graph.geometry_node_count = torch.tensor(
                    [graph.z.shape[0]], dtype=torch.long
                )
                # Preserve the historical joined fields for compatibility;
                # the v3 adapter consumes the explicit view fields above.
                graph.x = topology.x
                graph.edge_index = topology.edge_index
                graph.edge_attr = topology.edge_attr
                joined.append(graph)
            _save_packed_graphs(output_path, joined)
            item = {
                "path": relative,
                "source_gps": str(gps_path),
                "source_primary": str(primary_path),
                "rows": len(joined),
                "sha256": sha256_file(output_path),
            }
            existing[relative] = item
            manifest["files"] = [existing[key] for key in sorted(existing)]
            atomic_json(manifest_path, manifest)
            print(f"joined {relative}: {len(joined)} rows", flush=True)
    manifest["status"] = "complete"
    manifest["files"] = [existing[key] for key in sorted(existing)]
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
