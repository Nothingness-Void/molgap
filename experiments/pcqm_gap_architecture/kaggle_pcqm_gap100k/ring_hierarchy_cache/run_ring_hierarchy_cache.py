"""Kaggle CPU: deterministic smallest-ring hierarchy over accepted geometry."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_ring_hierarchy_cache_s42")
EXPECTED_RING_SOURCE_COMMIT = "58f425258031062c3c3762f13b7d4c160dffba65"
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
OFFICIAL_TRAIN_ROWS = 3_378_606


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
            "ogb==1.3.6",
            "torch-geometric==2.6.1",
        ]
    )


def source_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_ring.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one ring source tree/archive, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_ring_cache_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_ring.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected ring source archive layout: {modules}")
    return modules[0].parents[1]


def source_commit(root: Path) -> str:
    candidates = [
        root.parent / "PCQM_GAP100K_SOURCE_COMMIT.txt",
        root / "PCQM_GAP100K_SOURCE_COMMIT.txt",
    ]
    markers = [path for path in candidates if path.is_file()]
    if len(markers) != 1:
        raise RuntimeError(f"Expected one source marker, found {markers}")
    value = markers[0].read_text(encoding="utf-8").strip()
    if len(value) != 40:
        raise RuntimeError("Ring source marker is not a full commit hash")
    return value


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


def find_source_csv() -> Path:
    matches = list(Path("/kaggle/input").rglob("data.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one PCQM data.csv, found {matches}")
    return matches[0]


def selected_rows(root: Path, manifest: dict) -> set[int]:
    import torch

    selected = set()
    for shard in manifest["shards"]:
        graphs = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"Geometry graph count changed: {shard['file']}")
        selected.update(int(graph.row_index.view(-1)[0]) for graph in graphs)
    if len(selected) != 110_000:
        raise RuntimeError(f"Expected 110000 selected rows, found {len(selected)}")
    return selected


def selected_smiles(path: Path, selected: set[int]) -> tuple[dict[int, str], str]:
    import pandas as pd

    mapping: dict[int, str] = {}
    for frame in pd.read_csv(
        path,
        nrows=OFFICIAL_TRAIN_ROWS,
        usecols=["idx", "smiles"],
        chunksize=250_000,
    ):
        filtered = frame[frame["idx"].isin(selected)]
        mapping.update(
            (int(row.idx), str(row.smiles))
            for row in filtered.itertuples(index=False)
        )
    if set(mapping) != selected:
        raise RuntimeError(
            f"Selected PCQM SMILES are incomplete: {sorted(selected - set(mapping))[:20]}"
        )
    digest = hashlib.sha256()
    for row_index in sorted(mapping):
        digest.update(f"{row_index}\t{mapping[row_index]}\n".encode("utf-8"))
    return mapping, digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    try:
        install_dependencies()
        python_root = source_root()
        sys.path.insert(0, str(python_root))
        import torch

        from molgap.pcqm_ring import hierarchy_counts, with_ring_hierarchy

        commit = source_commit(python_root)
        if commit != EXPECTED_RING_SOURCE_COMMIT:
            raise RuntimeError(f"Ring source commit changed: {commit}")
        geometry_root, geometry_manifest = find_geometry_cache()
        selected = selected_rows(geometry_root, geometry_manifest)
        smiles_by_row, selected_smiles_sha256 = selected_smiles(
            find_source_csv(), selected
        )

        output_shards = []
        totals = {
            "train": {"graphs": 0, "rings": 0, "memberships": 0, "directed_relations": 0, "acyclic_graphs": 0},
            "validation": {"graphs": 0, "rings": 0, "memberships": 0, "directed_relations": 0, "acyclic_graphs": 0},
        }
        for ordinal, shard in enumerate(geometry_manifest["shards"]):
            role = shard["role"]
            parent_path = geometry_root / shard["file"]
            graphs = torch.load(parent_path, map_location="cpu", weights_only=False)
            converted = []
            for graph in graphs:
                row_index = int(graph.row_index.view(-1)[0])
                try:
                    converted.append(
                        with_ring_hierarchy(graph, smiles_by_row[row_index])
                    )
                except Exception as error:
                    failures.append(
                        {
                            "role": role,
                            "row_index": row_index,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
            if failures:
                atomic_json(
                    OUT / "ring_failures.json",
                    {
                        "format": "molgap-pcqm-gap100k-ring-hierarchy-failures-v1",
                        "failures": failures,
                    },
                )
                raise RuntimeError(
                    f"Ring extraction failed for {len(failures)} selected graphs"
                )
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            atomic_torch_save(output_path, converted)
            counts = hierarchy_counts(converted)
            acyclic = sum(int(graph.ring_features.shape[0] == 0) for graph in converted)
            record = {
                "role": role,
                "file": filename,
                "source_geometry_file": shard["file"],
                "graph_count": len(converted),
                "ring_count": counts["rings"],
                "membership_count": counts["memberships"],
                "directed_relation_count": counts["directed_relations"],
                "acyclic_graph_count": acyclic,
                "sha256": sha256_file(output_path),
            }
            output_shards.append(record)
            totals[role]["graphs"] += len(converted)
            totals[role]["rings"] += counts["rings"]
            totals[role]["memberships"] += counts["memberships"]
            totals[role]["directed_relations"] += counts["directed_relations"]
            totals[role]["acyclic_graphs"] += acyclic
            atomic_json(
                OUT / "progress.json",
                {
                    "format": "molgap-pcqm-gap100k-ring-hierarchy-progress-v1",
                    "complete": False,
                    "source_commit": commit,
                    "geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
                    "shards": output_shards,
                    "totals": totals,
                    "failure_count": 0,
                    "elapsed_s": time.perf_counter() - started,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            print(
                f"ring hierarchy {role} shard {ordinal}: graphs={len(converted)} "
                f"rings={counts['rings']} relations={counts['directed_relations']}",
                flush=True,
            )

        atomic_json(
            OUT / "ring_failures.json",
            {
                "format": "molgap-pcqm-gap100k-ring-hierarchy-failures-v1",
                "failures": [],
            },
        )
        aggregate = hashlib.sha256()
        for shard in output_shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        failures_path = OUT / "ring_failures.json"
        manifest = {
            "format": "molgap-pcqm-gap100k-ring-hierarchy-cache-v1",
            "complete": True,
            "source_commit": commit,
            "parent_geometry_source_commit": geometry_manifest["source_commit"],
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "official_train_rows_read": OFFICIAL_TRAIN_ROWS,
            "selected_smiles_sha256": selected_smiles_sha256,
            "ring_method": "RDKit-GetSymmSSSR-canonical-atom-tuples",
            "ring_feature_channels": 12,
            "ring_edge_feature_channels": 4,
            "ring_relations": ["spiro", "fused", "direct_bond", "conjugated_direct_bond"],
            "train_graphs": totals["train"]["graphs"],
            "validation_graphs": totals["validation"]["graphs"],
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
        summary = {
            "format": "molgap-pcqm-gap100k-ring-hierarchy-cache-run-v1",
            "complete": True,
            "source_commit": commit,
            "aggregate_sha256": manifest["aggregate_sha256"],
            "train_graphs": manifest["train_graphs"],
            "validation_graphs": manifest["validation_graphs"],
            "totals": totals,
            "elapsed_s": time.perf_counter() - started,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "failures": failures,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
