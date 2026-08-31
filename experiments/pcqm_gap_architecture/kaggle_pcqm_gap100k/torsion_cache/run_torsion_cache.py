"""Kaggle CPU derivation of the sparse torsion cache from accepted geometry."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_torsion_cache_s42")
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


def source_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_torsion.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"Expected one torsion source tree/archive, found {matches}/{archives}"
        )
    extracted = Path("/kaggle/working/_molgap_torsion_cache_source")
    import shutil

    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_torsion.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected torsion source archive layout: {modules}")
    return modules[0].parents[1]


def install_dependencies() -> None:
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.6.1",
        ]
    )


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
        raise RuntimeError(f"Expected one accepted geometry cache, found {candidates}")
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
    if float(manifest.get("valid_geometry_fraction", 0.0)) < 0.99:
        raise RuntimeError("Geometry cache valid fraction is below 0.99")
    failures_path = root / manifest["failures_file"]
    if sha256_file(failures_path) != manifest["failures_file_sha256"]:
        raise RuntimeError("Geometry failure ledger hash changed")
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


def load_geometry_failures(root: Path, manifest: dict) -> dict[int, dict]:
    payload = json.loads((root / manifest["failures_file"]).read_text(encoding="utf-8"))
    failures = payload.get("failures", [])
    return {int(row["row_index"]): row for row in failures}


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        install_dependencies()
        source_python_root = source_root()
        sys.path.insert(0, str(source_python_root))
        import torch

        from molgap.pcqm_torsion import with_torsion_cache

        source_marker = source_python_root.parent / "PCQM_GAP100K_SOURCE_COMMIT.txt"
        if not source_marker.is_file():
            source_marker = source_python_root / "PCQM_GAP100K_SOURCE_COMMIT.txt"
        if not source_marker.is_file():
            raise RuntimeError(f"Torsion source marker missing beside {source_python_root}")
        source_commit = source_marker.read_text(encoding="utf-8").strip()
        if len(source_commit) != 40:
            raise RuntimeError("Torsion source marker is not a full commit hash")
        geometry_root, geometry_manifest = find_geometry_cache()
        geometry_failures = load_geometry_failures(geometry_root, geometry_manifest)

        output_shards = []
        failures = []
        total_graphs = {"train": 0, "validation": 0}
        total_paths = {"train": 0, "validation": 0}
        total_valid_paths = {"train": 0, "validation": 0}
        invalid_geometry_rows = 0
        degenerate_paths = 0
        completed = {}
        progress_path = OUT / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("geometry_cache_aggregate_sha256") != EXPECTED_GEOMETRY_SHA256:
                raise RuntimeError("Torsion progress geometry identity changed")
            if progress.get("source_commit") != source_commit:
                raise RuntimeError("Torsion progress source identity changed")
            completed = {
                row["file"]: row for row in progress.get("shards", [])
            }
            failures = list(progress.get("failures", []))

        for ordinal, shard in enumerate(geometry_manifest["shards"]):
            role = shard["role"]
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            previous = completed.get(filename)
            if previous is not None and output_path.is_file():
                if sha256_file(output_path) != previous["sha256"]:
                    raise RuntimeError(f"Torsion shard changed during resume: {filename}")
                output_shards.append(previous)
                total_graphs[role] += int(previous["graph_count"])
                total_paths[role] += int(previous["torsion_path_count"])
                total_valid_paths[role] += int(previous["valid_torsion_path_count"])
                invalid_geometry_rows += int(previous["invalid_geometry_rows"])
                degenerate_paths += int(previous["degenerate_path_count"])
                continue

            payload = torch.load(
                geometry_root / shard["file"], map_location="cpu", weights_only=False
            )
            if len(payload) != int(shard["graph_count"]):
                raise RuntimeError(f"Geometry graph count changed: {shard['file']}")
            converted = []
            shard_paths = 0
            shard_valid_paths = 0
            shard_invalid_rows = 0
            shard_degenerate = 0
            for graph in payload:
                row_index = int(graph.row_index.view(-1)[0])
                converted_graph = with_torsion_cache(graph)
                path_count = int(converted_graph.torsion_edge_ids.shape[0])
                valid_count = int(converted_graph.torsion_valid.sum().item())
                shard_paths += path_count
                shard_valid_paths += valid_count
                if not bool(graph.geometry_valid.reshape(-1)[0]):
                    shard_invalid_rows += 1
                    failure = geometry_failures.get(row_index, {})
                    failures.append(
                        {
                            "role": role,
                            "row_index": row_index,
                            "reason": "parent_geometry_invalid",
                            "geometry_failure_type": failure.get("type"),
                            "geometry_failure_message": failure.get("message"),
                            "torsion_path_count": path_count,
                        }
                    )
                shard_degenerate += path_count - valid_count
                converted.append(converted_graph)
            atomic_torch_save(output_path, converted)
            record = {
                "role": role,
                "file": filename,
                "source_geometry_file": shard["file"],
                "graph_count": len(converted),
                "torsion_path_count": shard_paths,
                "valid_torsion_path_count": shard_valid_paths,
                "invalid_geometry_rows": shard_invalid_rows,
                "degenerate_path_count": shard_degenerate,
                "sha256": sha256_file(output_path),
            }
            output_shards.append(record)
            total_graphs[role] += len(converted)
            total_paths[role] += shard_paths
            total_valid_paths[role] += shard_valid_paths
            invalid_geometry_rows += shard_invalid_rows
            degenerate_paths += shard_degenerate
            atomic_json(
                progress_path,
                {
                    "format": "molgap-pcqm-gap100k-torsion-progress-v1",
                    "complete": False,
                    "source_commit": source_commit,
                    "geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
                    "shards": output_shards,
                    "train_graphs": total_graphs["train"],
                    "validation_graphs": total_graphs["validation"],
                    "torsion_paths": total_paths,
                    "valid_torsion_paths": total_valid_paths,
                    "invalid_geometry_rows": invalid_geometry_rows,
                    "degenerate_path_count": degenerate_paths,
                    "failures": failures,
                    "elapsed_s": time.perf_counter() - started,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            print(
                f"torsion {role} shard {ordinal}: graphs={len(converted)} "
                f"paths={shard_paths} valid={shard_valid_paths}",
                flush=True,
            )

        output_shards.sort(key=lambda row: (row["role"], row["file"]))
        if total_graphs != {"train": 100_000, "validation": 10_000}:
            raise RuntimeError(f"Torsion role counts are incomplete: {total_graphs}")
        failures_path = OUT / "torsion_failures.json"
        atomic_json(
            failures_path,
            {
                "format": "molgap-pcqm-gap100k-torsion-failures-v1",
                "policy": "retain-parent-geometry-rows-and-mask-invalid-paths",
                "failures": failures,
            },
        )
        aggregate = hashlib.sha256()
        for shard in output_shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        manifest = {
            "format": "molgap-pcqm-gap100k-etkdg-torsion-cache-v1",
            "complete": True,
            "source_commit": source_commit,
            "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "train_graphs": total_graphs["train"],
            "validation_graphs": total_graphs["validation"],
            "invalid_geometry_rows": invalid_geometry_rows,
            "torsion_paths": total_paths,
            "valid_torsion_paths": total_valid_paths,
            "degenerate_path_count": degenerate_paths,
            "torsion_definition": "directed_nonbacktracking_i_j_k_l",
            "torsion_edge_id_shape": ["num_torsions", 3],
            "torsion_wedge_id_shape": ["num_torsions", 2],
            "torsion_feature_definition": "[sin(phi), cos(phi), sin(2phi), cos(2phi)]",
            "torsion_feature_dtype": "float32",
            "invalid_geometry_policy": "zero_torsion_features_and_zero_path_mask",
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
        summary = {
            "format": "molgap-pcqm-gap100k-etkdg-torsion-cache-run-v1",
            "complete": True,
            "source_commit": source_commit,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "aggregate_sha256": manifest["aggregate_sha256"],
            "train_graphs": manifest["train_graphs"],
            "validation_graphs": manifest["validation_graphs"],
            "invalid_geometry_rows": manifest["invalid_geometry_rows"],
            "torsion_paths": manifest["torsion_paths"],
            "valid_torsion_paths": manifest["valid_torsion_paths"],
            "degenerate_path_count": manifest["degenerate_path_count"],
            "elapsed_s": time.perf_counter() - started,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "run_summary.json", summary)
        atomic_json(
            progress_path,
            {
                "format": "molgap-pcqm-gap100k-torsion-progress-v1",
                "complete": True,
                "source_commit": source_commit,
                "geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
                "shards": output_shards,
                "failures": failures,
                "elapsed_s": summary["elapsed_s"],
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "format": "molgap-pcqm-gap100k-torsion-cache-failure-v1",
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
