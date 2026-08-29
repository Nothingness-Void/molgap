"""Kaggle CPU: immutable ETKDGv3/MMFF geometry for PCQM Gap100K."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_geometry_cache_s42")
EXPECTED_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_PARENT_SOURCE_COMMIT = "35fadc9de63e22de7a1cfbe21e4f1af8888e075f"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
OFFICIAL_TRAIN_ROWS = 3_378_606
MINIMUM_VALID_FRACTION = 0.99
WORKERS = 4


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


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "rdkit==2025.3.5",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )


def source_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_geometry.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source tree/archive, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_geometry_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_geometry.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def find_wedge_cache() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-sparse-wedge-cache-v1":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one sparse-wedge cache, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "source_commit": EXPECTED_PARENT_SOURCE_COMMIT,
        "parent_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "aggregate_sha256": EXPECTED_WEDGE_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Wedge cache contract changed for {key}")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Wedge shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != EXPECTED_WEDGE_SHA256:
        raise RuntimeError("Wedge aggregate identity changed")
    return root, manifest


def find_source_csv() -> Path:
    matches = [
        path
        for path in Path("/kaggle/input").rglob("data.csv")
        if path.name == "data.csv"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one PCQM data.csv, found {matches}")
    return matches[0]


def selected_row_indices(root: Path, manifest: dict) -> set[int]:
    import torch

    selected = set()
    for shard in manifest["shards"]:
        graphs = torch.load(
            root / shard["file"], map_location="cpu", weights_only=False
        )
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"Wedge graph count changed: {shard['file']}")
        for graph in graphs:
            selected.add(int(graph.row_index.view(-1)[0]))
    if len(selected) != 110_000:
        raise RuntimeError(f"Expected 110000 distinct selected rows, found {len(selected)}")
    return selected


def load_selected_smiles(path: Path, selected: set[int]) -> tuple[dict[int, str], str]:
    import pandas as pd

    mapping: dict[int, str] = {}
    for frame in pd.read_csv(
        path,
        nrows=OFFICIAL_TRAIN_ROWS,
        usecols=["idx", "smiles"],
        chunksize=250_000,
    ):
        filtered = frame[frame["idx"].isin(selected)]
        for row in filtered.itertuples(index=False):
            mapping[int(row.idx)] = str(row.smiles)
    if set(mapping) != selected:
        missing = sorted(selected.difference(mapping))[:20]
        raise RuntimeError(f"Selected PCQM SMILES are incomplete: {missing}")
    digest = hashlib.sha256()
    for row_index in sorted(mapping):
        digest.update(f"{row_index}\t{mapping[row_index]}\n".encode("utf-8"))
    return mapping, digest.hexdigest()


def geometry_job(arguments):
    from molgap.pcqm_geometry import compute_etkdg_geometry

    return compute_etkdg_geometry(*arguments)


def attach_geometry(graph, result):
    import torch

    from molgap.pcqm_geometry import geometry_is_finite

    if not geometry_is_finite(result):
        raise RuntimeError("Geometry worker emitted non-finite values")
    graph.pos = torch.from_numpy(result.positions)
    graph.edge_distance = torch.from_numpy(result.edge_distance)
    graph.wedge_angle_cos = torch.from_numpy(result.wedge_angle_cos)
    graph.geometry_valid = torch.tensor(
        [1.0 if result.geometry_valid else 0.0], dtype=torch.float32
    )
    graph.mmff_converged = torch.tensor(
        [1.0 if result.mmff_converged else 0.0], dtype=torch.float32
    )
    return graph


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        install_dependencies()
        sys.path.insert(0, str(source_root()))
        import torch

        marker = next(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
        source_commit = marker.read_text(encoding="utf-8").strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"Geometry source commit changed: {source_commit}")
        parent_root, parent_manifest = find_wedge_cache()
        selected = selected_row_indices(parent_root, parent_manifest)
        smiles_by_row, selected_smiles_sha256 = load_selected_smiles(
            find_source_csv(), selected
        )

        output_shards = []
        failures = []
        stats = {
            "train": {"graphs": 0, "valid": 0, "mmff_converged": 0},
            "validation": {"graphs": 0, "valid": 0, "mmff_converged": 0},
        }
        distance_min = None
        distance_max = None
        angle_min = None
        angle_max = None
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(WORKERS, os.cpu_count() or 1)
        ) as executor:
            for ordinal, shard in enumerate(parent_manifest["shards"]):
                role = shard["role"]
                graphs = torch.load(
                    parent_root / shard["file"],
                    map_location="cpu",
                    weights_only=False,
                )
                jobs = []
                for graph in graphs:
                    row_index = int(graph.row_index.view(-1)[0])
                    jobs.append(
                        (
                            smiles_by_row[row_index],
                            row_index,
                            int(graph.num_nodes),
                            graph.edge_index.numpy(),
                            graph.wedge_edge_ids.numpy(),
                        )
                    )
                results = list(executor.map(geometry_job, jobs, chunksize=16))
                converted = []
                for graph, result in zip(graphs, results):
                    row_index = int(graph.row_index.view(-1)[0])
                    converted.append(attach_geometry(graph, result))
                    stats[role]["graphs"] += 1
                    stats[role]["valid"] += int(result.geometry_valid)
                    stats[role]["mmff_converged"] += int(result.mmff_converged)
                    if result.geometry_valid:
                        if result.edge_distance.size:
                            current_min = float(result.edge_distance.min())
                            current_max = float(result.edge_distance.max())
                            distance_min = (
                                current_min
                                if distance_min is None
                                else min(distance_min, current_min)
                            )
                            distance_max = (
                                current_max
                                if distance_max is None
                                else max(distance_max, current_max)
                            )
                        if result.wedge_angle_cos.size:
                            current_min = float(result.wedge_angle_cos.min())
                            current_max = float(result.wedge_angle_cos.max())
                            angle_min = (
                                current_min
                                if angle_min is None
                                else min(angle_min, current_min)
                            )
                            angle_max = (
                                current_max
                                if angle_max is None
                                else max(angle_max, current_max)
                            )
                    else:
                        failures.append(
                            {
                                "role": role,
                                "row_index": row_index,
                                "embed_attempt": result.embed_attempt,
                                "type": result.failure_type,
                                "message": result.failure_message,
                            }
                        )
                filename = f"{role}-{ordinal:04d}.pt"
                output_path = OUT / filename
                atomic_torch_save(output_path, converted)
                record = {
                    "role": role,
                    "file": filename,
                    "graph_count": len(converted),
                    "valid_geometry_count": sum(
                        int(result.geometry_valid) for result in results
                    ),
                    "sha256": sha256_file(output_path),
                }
                output_shards.append(record)
                atomic_json(
                    OUT / "progress.json",
                    {
                        "format": "molgap-pcqm-gap100k-geometry-progress-v1",
                        "complete": False,
                        "source_commit": source_commit,
                        "shards": output_shards,
                        "stats": stats,
                        "failure_count": len(failures),
                        "elapsed_s": time.perf_counter() - started,
                        "official_validation_role_read": False,
                        "test_dev_role_read": False,
                    },
                )
                print(
                    f"geometry {role} shard {ordinal}: graphs={len(converted)} "
                    f"valid={record['valid_geometry_count']}",
                    flush=True,
                )

        total_graphs = sum(item["graphs"] for item in stats.values())
        total_valid = sum(item["valid"] for item in stats.values())
        valid_fraction = total_valid / total_graphs
        atomic_json(
            OUT / "geometry_failures.json",
            {
                "format": "molgap-pcqm-gap100k-geometry-failures-v1",
                "failures": failures,
            },
        )
        if valid_fraction < MINIMUM_VALID_FRACTION:
            raise RuntimeError(
                f"Geometry valid fraction {valid_fraction:.6f} is below "
                f"{MINIMUM_VALID_FRACTION:.6f}"
            )
        aggregate = hashlib.sha256()
        for shard in output_shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        failures_path = OUT / "geometry_failures.json"
        manifest = {
            "format": "molgap-pcqm-gap100k-etkdg-geometry-cache-v1",
            "complete": True,
            "source_commit": source_commit,
            "parent_cache_source_commit": parent_manifest["source_commit"],
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
            "official_train_rows_read": OFFICIAL_TRAIN_ROWS,
            "selected_smiles_sha256": selected_smiles_sha256,
            "train_graphs": stats["train"]["graphs"],
            "validation_graphs": stats["validation"]["graphs"],
            "valid_geometry_graphs": total_valid,
            "invalid_geometry_graphs": total_graphs - total_valid,
            "valid_geometry_fraction": valid_fraction,
            "mmff_converged_graphs": sum(
                item["mmff_converged"] for item in stats.values()
            ),
            "geometry_method": "ETKDGv3",
            "geometry_seed_policy": "row-index-derived-base42",
            "optimization_method": "MMFF94s",
            "optimization_max_iters": 200,
            "single_conformer": True,
            "distance_feature": "directed-real-bond-length-angstrom",
            "angle_feature": "directed-nonbacktracking-wedge-cosine",
            "distance_min": distance_min,
            "distance_max": distance_max,
            "angle_cos_min": angle_min,
            "angle_cos_max": angle_max,
            "failures_file": failures_path.name,
            "failures_file_sha256": sha256_file(failures_path),
            "shards": output_shards,
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
                "format": "molgap-pcqm-gap100k-etkdg-geometry-run-v1",
                "complete": True,
                "source_commit": source_commit,
                "aggregate_sha256": manifest["aggregate_sha256"],
                "valid_geometry_fraction": valid_fraction,
                "elapsed_s": time.perf_counter() - started,
                "gpu_used": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        atomic_json(
            OUT / "progress.json",
            {
                "format": "molgap-pcqm-gap100k-geometry-progress-v1",
                "complete": True,
                "source_commit": source_commit,
                "shards": output_shards,
                "stats": stats,
                "failure_count": len(failures),
                "elapsed_s": time.perf_counter() - started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
