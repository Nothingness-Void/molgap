"""Kaggle CPU: deterministic normalized-Laplacian eigenpairs for PCQM 100K."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_signnet_lappe_cache_s42")
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
LAPPE_DIM = 8
POSITIVE_EIGENVALUE_EPS = 1.0e-8


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
        [sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.6.1"]
    )


def source_commit() -> str:
    markers = list(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
    if len(markers) != 1:
        raise RuntimeError(f"Expected one source marker, found {markers}")
    value = markers[0].read_text(encoding="utf-8").strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError("LapPE source marker is not a full Git hash")
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


def attach_lappe(graph):
    import torch

    node_count = int(graph.num_nodes)
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float64)
    if graph.edge_index.shape[1]:
        source, target = graph.edge_index.long()
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1)
    inverse_sqrt = torch.where(
        degree > 0,
        degree.clamp_min(1.0).rsqrt(),
        torch.zeros_like(degree),
    )
    laplacian = torch.eye(node_count, dtype=torch.float64)
    laplacian = laplacian - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    positive = torch.nonzero(
        eigenvalues > POSITIVE_EIGENVALUE_EPS,
        as_tuple=False,
    ).view(-1)
    selected = positive[:LAPPE_DIM]
    available = int(selected.numel())
    eigvec = torch.zeros((node_count, LAPPE_DIM), dtype=torch.float32)
    eigval = torch.zeros((node_count, LAPPE_DIM), dtype=torch.float32)
    mask = torch.zeros((node_count, LAPPE_DIM), dtype=torch.bool)
    if available:
        values = eigenvalues[selected].to(torch.float32)
        eigvec[:, :available] = eigenvectors[:, selected].to(torch.float32)
        eigval[:, :available] = values.view(1, -1).expand(node_count, -1)
        mask[:, :available] = True
    if not torch.isfinite(eigvec).all() or not torch.isfinite(eigval).all():
        raise RuntimeError("Laplacian eigendecomposition produced non-finite values")
    graph.lap_eigvec = eigvec
    graph.lap_eigval = eigval
    graph.lap_mask = mask
    return graph, available, int((eigenvalues <= POSITIVE_EIGENVALUE_EPS).sum())


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        install_dependencies()
        commit = source_commit()
        root, parent = find_geometry_cache()
        shards = []
        role_graphs = {"train": 0, "validation": 0}
        role_valid_modes = {"train": 0, "validation": 0}
        disconnected_graphs = {"train": 0, "validation": 0}
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
            valid_modes = 0
            disconnected = 0
            for graph in graphs:
                graph, available, zero_modes = attach_lappe(graph)
                converted.append(graph)
                valid_modes += available
                disconnected += int(zero_modes > 1)
            filename = f"{role}-{ordinal:04d}.pt"
            output_path = OUT / filename
            atomic_torch_save(output_path, converted)
            record = {
                "role": role,
                "file": filename,
                "graph_count": len(converted),
                "valid_eigenmodes": valid_modes,
                "disconnected_graphs": disconnected,
                "finite_values": True,
                "sha256": sha256_file(output_path),
            }
            shards.append(record)
            role_graphs[role] += len(converted)
            role_valid_modes[role] += valid_modes
            disconnected_graphs[role] += disconnected
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
                f"lappe {role} shard {ordinal}: graphs={len(converted)} "
                f"valid_modes={valid_modes}",
                flush=True,
            )

        if role_graphs != {"train": 100_000, "validation": 10_000}:
            raise RuntimeError(f"LapPE role counts changed: {role_graphs}")
        aggregate = hashlib.sha256()
        for shard in shards:
            aggregate.update(
                f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                    "ascii"
                )
            )
        manifest = {
            "format": "molgap-pcqm-gap100k-lappe-cache-v1",
            "complete": True,
            "source_commit": commit,
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "train_graphs": role_graphs["train"],
            "validation_graphs": role_graphs["validation"],
            "laplacian": "symmetric-normalized-unweighted-covalent",
            "eigenpair_policy": "lowest-8-nontrivial-zero-padded",
            "positive_eigenvalue_epsilon": POSITIVE_EIGENVALUE_EPS,
            "lappe_dim": LAPPE_DIM,
            "role_valid_eigenmodes": role_valid_modes,
            "disconnected_graphs": disconnected_graphs,
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
                "format": "molgap-pcqm-gap100k-lappe-cache-run-v1",
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
        atomic_json(OUT / "progress.json", {**manifest, "elapsed_s": time.perf_counter() - started})
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
