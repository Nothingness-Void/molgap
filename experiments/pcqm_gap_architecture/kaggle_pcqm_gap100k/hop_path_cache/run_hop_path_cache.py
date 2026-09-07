"""Kaggle CPU: exact-shortest two/three-hop path cache for PCQM 100K."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_hop_path_cache_s42")
EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_PARENT_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str:
    markers = list(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
    if len(markers) != 1:
        raise RuntimeError(f"Expected one source marker, found {markers}")
    value = markers[0].read_text(encoding="utf-8").strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError("Hop-path source marker is not a full Git hash")
    return value


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("molgap/pcqm_hop_path.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source tree/archive, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_hop_path_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_hop_path.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def find_geometry_cache() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-etkdg-geometry-cache-v1":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one geometry cache, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
        "aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "single_conformer": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Geometry cache contract changed for {key}")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Geometry shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError("Geometry aggregate identity changed")
    return root, manifest


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.6.1"]
        )
        commit = source_commit()
        sys.path.insert(0, str(source_python_root()))
        from molgap.pcqm_hop_path import attach_hop_path_features

        root, parent = find_geometry_cache()
        shards = []
        role_graphs = {"train": 0, "validation": 0}
        role_hop2 = {"train": 0, "validation": 0}
        role_hop3 = {"train": 0, "validation": 0}
        role_multipath = {"train": 0, "validation": 0}
        maximum_path_count = 0
        for ordinal, parent_shard in enumerate(parent["shards"]):
            import torch

            role = parent_shard["role"]
            graphs = torch.load(
                root / parent_shard["file"],
                map_location="cpu",
                weights_only=False,
            )
            if len(graphs) != int(parent_shard["graph_count"]):
                raise RuntimeError(f"Parent graph count changed: {parent_shard['file']}")
            converted = []
            shard_hop2 = 0
            shard_hop3 = 0
            shard_multipath = 0
            shard_maximum = 0
            for graph in graphs:
                graph, stats = attach_hop_path_features(graph)
                converted.append(graph)
                shard_hop2 += stats["hop2_relations"]
                shard_hop3 += stats["hop3_relations"]
                shard_multipath += stats["multipath_relations"]
                shard_maximum = max(shard_maximum, stats["maximum_path_count"])
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            atomic_torch_save(output_path, converted)
            record = {
                "role": role,
                "file": filename,
                "graph_count": len(converted),
                "hop2_relations": shard_hop2,
                "hop3_relations": shard_hop3,
                "multipath_relations": shard_multipath,
                "maximum_path_count": shard_maximum,
                "finite_values": True,
                "sha256": sha256_file(output_path),
            }
            shards.append(record)
            role_graphs[role] += len(converted)
            role_hop2[role] += shard_hop2
            role_hop3[role] += shard_hop3
            role_multipath[role] += shard_multipath
            maximum_path_count = max(maximum_path_count, shard_maximum)
            atomic_json(
                OUT / "progress.json",
                {
                    "complete": False,
                    "source_commit": commit,
                    "shards": shards,
                    "role_graphs": role_graphs,
                    "elapsed_s": time.perf_counter() - started,
                    "gpu_used": False,
                    "model_inference_executed": False,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            print(
                f"hop-path {role} shard {ordinal}: graphs={len(converted)} "
                f"hop2={shard_hop2} hop3={shard_hop3}",
                flush=True,
            )

        if role_graphs != {"train": 100_000, "validation": 10_000}:
            raise RuntimeError(f"Hop-path role counts changed: {role_graphs}")
        if min(role_hop2.values()) <= 0 or min(role_hop3.values()) <= 0:
            raise RuntimeError("Hop-path cache contains an empty role")
        aggregate = hashlib.sha256()
        for shard in shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        manifest = {
            "format": "molgap-pcqm-gap100k-hop-path-cache-v1",
            "complete": True,
            "source_commit": commit,
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "train_graphs": role_graphs["train"],
            "validation_graphs": role_graphs["validation"],
            "hops": [2, 3],
            "feature_channels": 8,
            "feature_definition": (
                "hop2,hop3,log1p_path_count,mean_single,mean_double,"
                "mean_triple,mean_aromatic,fully_conjugated_fraction"
            ),
            "exact_shortest_path": True,
            "simple_paths": True,
            "directed_relations": True,
            "role_hop2_relations": role_hop2,
            "role_hop3_relations": role_hop3,
            "role_multipath_relations": role_multipath,
            "maximum_path_count": maximum_path_count,
            "failure_count": 0,
            "shards": shards,
            "aggregate_sha256": aggregate.hexdigest(),
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "manifest.json", manifest)
        atomic_json(
            OUT / "run_summary.json",
            {
                "format": "molgap-pcqm-gap100k-hop-path-cache-run-v1",
                "complete": True,
                "source_commit": commit,
                "aggregate_sha256": manifest["aggregate_sha256"],
                "elapsed_s": time.perf_counter() - started,
                "gpu_used": False,
                "model_inference_executed": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        atomic_json(
            OUT / "progress.json",
            {**manifest, "elapsed_s": time.perf_counter() - started},
        )
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
                "gpu_used": False,
                "model_inference_executed": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
