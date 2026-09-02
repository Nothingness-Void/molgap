"""Kaggle CPU: deterministic non-covalent contacts over accepted geometry."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_contactstate_cache_s42")
EXPECTED_SOURCE_COMMIT = "7f2f8ce476f654320f07e2c2e630f473d7d81c72"
EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_GRAPH_SHA256 = "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
EXPECTED_WEDGE_SHA256 = "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
EXPECTED_GEOMETRY_SHA256 = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
MAX_DIRECTED_EDGES = 10_000_000


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
        [sys.executable, "-m", "pip", "install", "-q", "ogb==1.3.6", "torch-geometric==2.6.1"]
    )


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("molgap/pcqm_contact.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one ContactState source, found {modules}/{archives}")
    extracted = Path("/kaggle/working/_molgap_contactstate_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_contact.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected ContactState source layout: {modules}")
    return modules[0].parents[1]


def source_commit(root: Path) -> str:
    markers = [
        path
        for path in (
            root.parent / "PCQM_GAP100K_SOURCE_COMMIT.txt",
            root / "PCQM_GAP100K_SOURCE_COMMIT.txt",
        )
        if path.is_file()
    ]
    if len(markers) != 1:
        raise RuntimeError(f"Expected one source marker, found {markers}")
    return markers[0].read_text(encoding="utf-8").strip()


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
        "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
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
        aggregate.update(f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii"))
    if aggregate.hexdigest() != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError("Geometry aggregate identity changed")
    return root, manifest


def empty_totals() -> dict:
    return {
        "graphs": 0,
        "valid_geometry_graphs": 0,
        "invalid_geometry_graphs": 0,
        "graphs_with_contacts": 0,
        "atoms_with_contacts": 0,
        "undirected_pairs": 0,
        "directed_edges": 0,
        "cross_component_directed_edges": 0,
        "maximum_directed_edges_per_graph": 0,
        "atom_type_pairs": {},
    }


def merge_stats(total: dict, part: dict) -> None:
    for key in (
        "graphs", "valid_geometry_graphs", "invalid_geometry_graphs",
        "graphs_with_contacts", "atoms_with_contacts", "undirected_pairs",
        "directed_edges", "cross_component_directed_edges",
    ):
        total[key] += int(part[key])
    total["maximum_directed_edges_per_graph"] = max(
        total["maximum_directed_edges_per_graph"],
        int(part["maximum_directed_edges_per_graph"]),
    )
    pairs = Counter(total["atom_type_pairs"])
    pairs.update(part["atom_type_pairs"])
    total["atom_type_pairs"] = dict(sorted(pairs.items()))


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    try:
        install_dependencies()
        python_root = source_root()
        sys.path.insert(0, str(python_root))
        import torch
        from molgap.pcqm_contact import (
            CONTACT_CUTOFF_ANGSTROM,
            EXCLUDED_COVALENT_HOPS,
            contact_contract_violations,
            contact_statistics,
            with_non_covalent_contacts,
        )

        commit = source_commit(python_root)
        if commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"ContactState source commit changed: {commit}")
        geometry_root, geometry_manifest = find_geometry_cache()
        output_shards = []
        totals = {"train": empty_totals(), "validation": empty_totals()}
        global_distance_min = None
        global_distance_max = None

        for ordinal, shard in enumerate(geometry_manifest["shards"]):
            role = shard["role"]
            graphs = torch.load(geometry_root / shard["file"], map_location="cpu", weights_only=False)
            if len(graphs) != int(shard["graph_count"]):
                raise RuntimeError(f"Geometry graph count changed: {shard['file']}")
            converted = []
            for graph in graphs:
                row_index = int(graph.row_index.view(-1)[0])
                try:
                    converted_graph = with_non_covalent_contacts(graph)
                    violations = contact_contract_violations(converted_graph)
                    if violations:
                        raise ValueError(",".join(violations))
                    converted.append(converted_graph)
                except Exception as error:
                    failures.append({
                        "role": role,
                        "row_index": row_index,
                        "type": type(error).__name__,
                        "message": str(error),
                    })
            if failures:
                atomic_json(OUT / "contact_failures.json", {"failures": failures})
                raise RuntimeError(f"Contact conversion failed for {len(failures)} graphs")
            part = contact_statistics(converted)
            merge_stats(totals[role], part)
            distances = [graph.contact_distance.view(-1) for graph in converted if graph.contact_distance.numel()]
            if distances:
                joined = torch.cat(distances)
                current_min = float(joined.min())
                current_max = float(joined.max())
                global_distance_min = current_min if global_distance_min is None else min(global_distance_min, current_min)
                global_distance_max = current_max if global_distance_max is None else max(global_distance_max, current_max)
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            atomic_torch_save(output_path, converted)
            record = {
                "role": role,
                "file": filename,
                "source_geometry_file": shard["file"],
                "graph_count": len(converted),
                "statistics": part,
                "sha256": sha256_file(output_path),
            }
            output_shards.append(record)
            atomic_json(OUT / "progress.json", {
                "format": "molgap-pcqm-gap100k-contactstate-progress-v1",
                "complete": False,
                "source_commit": commit,
                "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
                "shards": output_shards,
                "totals": totals,
                "failure_count": 0,
                "elapsed_s": time.perf_counter() - started,
                "gpu_used": False,
                "model_inference_executed": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            })
            print(f"contacts {role} shard {ordinal}: graphs={len(converted)} directed={part['directed_edges']}", flush=True)

        directed_total = totals["train"]["directed_edges"] + totals["validation"]["directed_edges"]
        if directed_total > MAX_DIRECTED_EDGES:
            raise RuntimeError(f"Contact cache exceeds edge budget: {directed_total}")
        atomic_json(OUT / "contact_failures.json", {"format": "molgap-pcqm-gap100k-contactstate-failures-v1", "failures": []})
        failures_path = OUT / "contact_failures.json"
        aggregate = hashlib.sha256()
        for shard in output_shards:
            aggregate.update(f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii"))
        manifest = {
            "format": "molgap-pcqm-gap100k-contactstate-cache-v1",
            "complete": True,
            "source_commit": commit,
            "parent_geometry_source_commit": geometry_manifest["source_commit"],
            "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "contact_method": "ETKDGv3+MMFF94s-distance-cutoff-exclude-covalent-hops",
            "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
            "excluded_covalent_hops": EXCLUDED_COVALENT_HOPS,
            "directed_storage": True,
            "neighbor_cap": None,
            "train_graphs": totals["train"]["graphs"],
            "validation_graphs": totals["validation"]["graphs"],
            "distance_min_angstrom": global_distance_min,
            "distance_max_angstrom": global_distance_max,
            "directed_edge_budget": MAX_DIRECTED_EDGES,
            "totals": totals,
            "failure_count": 0,
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
        atomic_json(OUT / "run_summary.json", {
            "format": "molgap-pcqm-gap100k-contactstate-cache-run-v1",
            "complete": True,
            "source_commit": commit,
            "aggregate_sha256": manifest["aggregate_sha256"],
            "totals": totals,
            "distance_min_angstrom": global_distance_min,
            "distance_max_angstrom": global_distance_max,
            "elapsed_s": time.perf_counter() - started,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        })
        print(json.dumps(json.loads((OUT / "run_summary.json").read_text()), indent=2), flush=True)
    except Exception as error:
        atomic_json(OUT / "failure.json", {
            "type": type(error).__name__,
            "message": str(error),
            "failures": failures,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        })
        raise


if __name__ == "__main__":
    main()
