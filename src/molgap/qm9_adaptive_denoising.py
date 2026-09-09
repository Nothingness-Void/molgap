"""Track-C QM9 screen for local fixed versus adaptive geometry denoising.

The reusable implementation lives here; Kaggle entry points remain thin.  The
module deliberately imports torch only inside executable functions so cache and
result acceptance can inspect manifests without constructing a model.  Role
identities come from the accepted ``qm9_gape``/``qm9_local_hierarchy`` cache.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from .screen_policy import validate_paired_screen_contract, validate_screen_arm


TRAIN_ROWS = 30_000
VALIDATION_ROWS = 3_000
TOTAL_SELECTED_ROWS = 33_000
EXPECTED_SOURCE_ROWS = 130_831
EXPECTED_RDKIT_VERSION = "2023.09.6"
EXPECTED_PROCESSED_SHA256 = (
    "90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721"
)
EXPECTED_RAW_SDF_SHA256 = (
    "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
)
SPLIT_FINGERPRINT = "62f1cdefdaec6877"
PARENT_CACHE_AGGREGATE_SHA256 = (
    "80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340"
)
PARENT_CANONICAL_VALIDITY_SHA256 = (
    "a0882b5ec0e465772cb0f74b687a7a985ef8c89409f36c922606686f5f0d76b8"
)
ATOM_FEATURE_CHANNELS = 9
BOND_FEATURE_CHANNELS = 3
RWSE_DIM = 16
HIDDEN_CHANNELS = 192
NUM_LAYERS = 9
EDGE_STATE_CHANNELS = 64
WEDGE_CHANNELS = 16
GEOMETRY_BASIS_CHANNELS = 16
BATCH_SIZE = 128
SEED = 42
SCRATCH_EPOCHS = 40
PRETRAIN_EPOCHS = 10
GAP_EPOCHS = 30
TOTAL_ENCODER_EPOCHS = 40
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
MIN_LEARNING_RATE = 1e-6
DENOISING_SIGMA = 0.1
DENOISING_PRIOR_SIGMA = 0.1
KL_WEIGHT = 1.0
ADAPTIVE_LOG_SCALE_RADIUS = 0.5
MIN_GEOMETRY_VALID_FRACTION = 0.99
MIN_GAIN_VS_SCRATCH_EV = 0.003
MIN_GAIN_VS_FIXED_EV = 0.001
TASK_ID = "qm9-adaptive-local-denoising-s42-v1"
PLATFORM_ID = "kaggle2"


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


def state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_shards(shards: list[dict]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _find_parent_cache(root: Path) -> tuple[Path, dict]:
    candidates = []
    for manifest_path in root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            manifest.get("format")
            == "molgap-qm9-edgestate-local-hierarchy-cache-v2"
            and manifest.get("aggregate_sha256") == PARENT_CACHE_AGGREGATE_SHA256
        ):
            candidates.append((manifest_path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one frozen GAPE parent cache, found {candidates}")
    return candidates[0]


def _attach_geometry(graph, result):
    import torch

    from .pcqm_geometry import geometry_is_finite

    if not geometry_is_finite(result):
        raise RuntimeError("Geometry worker emitted non-finite values")
    graph.pos = torch.from_numpy(result.positions)
    graph.edge_distance = torch.from_numpy(result.edge_distance)
    graph.wedge_angle_cos = torch.from_numpy(result.wedge_angle_cos)
    graph.geometry_valid = torch.tensor(
        [1.0 if result.geometry_valid else 0.0], dtype=torch.float32
    )
    graph.geometry_failure_mask = torch.tensor(
        [0 if result.geometry_valid else 1], dtype=torch.uint8
    )
    graph.geometry_failure_type = str(result.failure_type or "")
    graph.mmff_converged = torch.tensor(
        [1.0 if result.mmff_converged else 0.0], dtype=torch.float32
    )
    return graph


def _geometry_job(arguments):
    from .pcqm_geometry import compute_etkdg_geometry

    return compute_etkdg_geometry(*arguments)


def _verify_parent_graph_identity(graph, smiles: str) -> None:
    """Prove that the regenerated canonical molecule is the accepted 2D graph."""
    from ogb.utils.mol import smiles2graph

    payload = smiles2graph(smiles)
    checks = {
        "node_feat": np.array_equal(
            np.asarray(payload["node_feat"], dtype=np.int64), graph.x.numpy()
        ),
        "edge_index": np.array_equal(
            np.asarray(payload["edge_index"], dtype=np.int64),
            graph.edge_index.numpy(),
        ),
        "edge_feat": np.array_equal(
            np.asarray(payload["edge_feat"], dtype=np.int64),
            graph.edge_attr.numpy(),
        ),
    }
    if not all(checks.values()):
        row_id = int(graph.row_id.view(-1)[0])
        raise RuntimeError(f"Parent graph identity changed at row {row_id}: {checks}")


def build_cache(
    parent_search_root: Path,
    output_root: Path,
    *,
    source_commit: str,
) -> dict:
    """Derive train/validation-only wedge and deterministic ETKDG geometry."""
    import torch
    from rdkit import Chem, RDLogger, rdBase

    from .pcqm_wedge import with_wedge_cache
    from .qm9_data import load_qm9_records, prepare_qm9_files
    from .qm9_local_hierarchy import _record_molecule, load_cache

    output_root.mkdir(parents=True, exist_ok=True)
    final_manifest = output_root / "manifest.json"
    if final_manifest.is_file():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            return existing
        raise RuntimeError("Incomplete final geometry manifest already exists")

    parent_root, parent_manifest = _find_parent_cache(parent_search_root)
    roles, verified_parent = load_cache(
        parent_root, PARENT_CACHE_AGGREGATE_SHA256
    )
    if (
        verified_parent.get("split_fingerprint") != SPLIT_FINGERPRINT
        or verified_parent.get("canonical_validity_sha256")
        != PARENT_CANONICAL_VALIDITY_SHA256
    ):
        raise RuntimeError("Frozen parent role identity changed")
    validity_source = parent_root / "canonical_validity.json"
    shutil.copy2(validity_source, output_root / "canonical_validity.json")

    source_workspace = output_root.parent / "_qm9_adaptive_source"
    files = prepare_qm9_files(source_workspace)
    records = load_qm9_records(source_workspace)
    source_checks = {
        "source_rows": len(records) == EXPECTED_SOURCE_ROWS,
        "rdkit_version": rdBase.rdkitVersion == EXPECTED_RDKIT_VERSION,
        "processed_sha256": sha256_file(files["processed"])
        == EXPECTED_PROCESSED_SHA256,
        "raw_sdf_sha256": sha256_file(files["raw_sdf"])
        == EXPECTED_RAW_SDF_SHA256,
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"Frozen QM9 source contract changed: {source_checks}")
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(
        str(files["raw_sdf"]), removeHs=False, sanitize=False
    )

    progress_path = output_root / "progress.json"
    completed_by_file = {}
    failures = []
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_required = {
            "format": "molgap-qm9-adaptive-denoising-cache-progress-v1",
            "source_commit": source_commit,
            "parent_cache_aggregate_sha256": PARENT_CACHE_AGGREGATE_SHA256,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "test_role_read": False,
        }
        if any(progress.get(key) != value for key, value in progress_required.items()):
            raise RuntimeError("Cache progress contract changed")
        failures = list(progress.get("failures", []))
        completed_by_file = {
            shard["file"]: shard for shard in progress.get("completed_shards", [])
        }
    shards = []
    stats = {
        "train": {"graphs": 0, "valid": 0, "wedges": 0},
        "validation": {"graphs": 0, "valid": 0, "wedges": 0},
    }
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(4, os.cpu_count() or 1)
    ) as executor:
        for role in ("train", "validation"):
            role_graphs = roles[role]
            for part_number, start in enumerate(range(0, len(role_graphs), 2_000)):
                source_graphs = role_graphs[start : start + 2_000]
                filename = f"{role}_part_{part_number:03d}.pt"
                path = output_root / filename
                if filename in completed_by_file:
                    shard = completed_by_file[filename]
                    if (
                        shard.get("role") != role
                        or not path.is_file()
                        or sha256_file(path) != shard.get("sha256")
                    ):
                        raise RuntimeError(f"Completed cache shard changed: {filename}")
                    converted = torch.load(path, map_location="cpu", weights_only=False)
                    expected_ids = [
                        int(graph.row_id.view(-1)[0]) for graph in source_graphs
                    ]
                    observed_ids = [
                        int(graph.row_id.view(-1)[0]) for graph in converted
                    ]
                    if (
                        observed_ids != expected_ids
                        or len(converted) != shard.get("graph_count")
                    ):
                        raise RuntimeError(f"Completed shard identity changed: {filename}")
                    stats[role]["graphs"] += len(converted)
                    stats[role]["valid"] += sum(
                        int(graph.geometry_valid.item()) for graph in converted
                    )
                    stats[role]["wedges"] += sum(
                        int(graph.wedge_edge_ids.shape[0]) for graph in converted
                    )
                    shards.append(shard)
                    print(f"geometry {role} part {part_number:03d}: resumed", flush=True)
                    continue
                if path.exists():
                    raise RuntimeError(f"Untracked cache shard exists: {filename}")
                wedge_graphs = []
                jobs = []
                for graph in source_graphs:
                    row_id = int(graph.row_id.view(-1)[0])
                    _molecule, smiles = _record_molecule(records[row_id], supplier)
                    _verify_parent_graph_identity(graph, smiles)
                    graph = with_wedge_cache(graph)
                    wedge_graphs.append(graph)
                    jobs.append(
                        (
                            smiles,
                            row_id,
                            int(graph.num_nodes),
                            graph.edge_index.numpy(),
                            graph.wedge_edge_ids.numpy(),
                        )
                    )
                results = list(executor.map(_geometry_job, jobs, chunksize=16))
                converted = []
                for graph, result in zip(wedge_graphs, results):
                    row_id = int(graph.row_id.view(-1)[0])
                    graph = _attach_geometry(graph, result)
                    converted.append(graph)
                    stats[role]["graphs"] += 1
                    stats[role]["valid"] += int(result.geometry_valid)
                    stats[role]["wedges"] += int(graph.wedge_edge_ids.shape[0])
                    if not result.geometry_valid:
                        failures.append(
                            {
                                "role": role,
                                "row_id": row_id,
                                "embed_attempt": result.embed_attempt,
                                "type": result.failure_type,
                                "message": result.failure_message,
                            }
                        )
                atomic_torch_save(path, converted)
                shard = {
                    "role": role,
                    "file": filename,
                    "graph_count": len(converted),
                    "valid_geometry_count": sum(
                        int(graph.geometry_valid.item()) for graph in converted
                    ),
                    "wedge_count": sum(
                        int(graph.wedge_edge_ids.shape[0]) for graph in converted
                    ),
                    "sha256": sha256_file(path),
                }
                shards.append(shard)
                atomic_json(
                    progress_path,
                    {
                        "format": "molgap-qm9-adaptive-denoising-cache-progress-v1",
                        "complete": False,
                        "source_commit": source_commit,
                        "parent_cache_aggregate_sha256": PARENT_CACHE_AGGREGATE_SHA256,
                        "split_fingerprint": SPLIT_FINGERPRINT,
                        "completed_shards": shards,
                        "failure_count": len(failures),
                        "failures": failures,
                        "test_role_read": False,
                    },
                )
                print(
                    f"geometry {role} part {part_number:03d}: "
                    f"graphs={len(converted)} valid={shard['valid_geometry_count']}",
                    flush=True,
                )

    atomic_json(
        output_root / "geometry_failures.json",
        {
            "format": "molgap-qm9-adaptive-denoising-geometry-failures-v1",
            "failures": failures,
        },
    )
    if {role: stats[role]["graphs"] for role in stats} != {
        "train": TRAIN_ROWS,
        "validation": VALIDATION_ROWS,
    }:
        raise RuntimeError("Geometry role counts changed")
    valid_geometry_count = sum(stats[role]["valid"] for role in stats)
    valid_geometry_fraction = valid_geometry_count / TOTAL_SELECTED_ROWS
    if valid_geometry_fraction < MIN_GEOMETRY_VALID_FRACTION:
        raise RuntimeError(
            f"Geometry valid fraction {valid_geometry_fraction:.6f} is below "
            f"{MIN_GEOMETRY_VALID_FRACTION:.6f}"
        )
    aggregate_sha256 = _aggregate_shards(shards)
    manifest = {
        "format": "molgap-qm9-adaptive-denoising-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "parent_cache_aggregate_sha256": PARENT_CACHE_AGGREGATE_SHA256,
        "parent_manifest_sha256": sha256_file(parent_root / "manifest.json"),
        "canonical_validity_sha256": sha256_file(
            output_root / "canonical_validity.json"
        ),
        "processed_source_sha256": sha256_file(files["processed"]),
        "raw_sdf_sha256": sha256_file(files["raw_sdf"]),
        "split_fingerprint": SPLIT_FINGERPRINT,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "atom_feature_channels": ATOM_FEATURE_CHANNELS,
        "bond_feature_channels": BOND_FEATURE_CHANNELS,
        "rwse_channels": RWSE_DIM,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "single_conformer": True,
        "dft_coordinates_used": False,
        "geometry_failure_mask": True,
        "geometry_failure_type": True,
        "stats": stats,
        "failure_count": len(failures),
        "valid_geometry_count": valid_geometry_count,
        "valid_geometry_fraction": valid_geometry_fraction,
        "geometry_failures_sha256": sha256_file(
            output_root / "geometry_failures.json"
        ),
        "shards": shards,
        "aggregate_sha256": aggregate_sha256,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(final_manifest, manifest)
    atomic_json(
        progress_path,
        {
            "format": "molgap-qm9-adaptive-denoising-cache-progress-v1",
            "complete": True,
            "source_commit": source_commit,
            "parent_cache_aggregate_sha256": PARENT_CACHE_AGGREGATE_SHA256,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "completed_shards": shards,
            "failure_count": len(failures),
            "failures": failures,
            "test_role_read": False,
        },
    )
    atomic_json(
        output_root / "run_summary.json",
        {
            "format": "molgap-qm9-adaptive-denoising-cache-run-v1",
            "complete": True,
            "source_commit": source_commit,
            "aggregate_sha256": aggregate_sha256,
            "gpu_used": False,
            "model_inference_executed": False,
            "official_pcqm_roles_read": False,
            "test_role_read": False,
        },
    )
    return manifest


def verify_cache(cache_root: Path, expected_sha256: str | None = None) -> dict:
    import torch

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-adaptive-denoising-cache-v1",
        "complete": True,
        "parent_cache_aggregate_sha256": PARENT_CACHE_AGGREGATE_SHA256,
        "canonical_validity_sha256": PARENT_CANONICAL_VALIDITY_SHA256,
        "processed_source_sha256": EXPECTED_PROCESSED_SHA256,
        "raw_sdf_sha256": EXPECTED_RAW_SDF_SHA256,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "atom_feature_channels": ATOM_FEATURE_CHANNELS,
        "bond_feature_channels": BOND_FEATURE_CHANNELS,
        "rwse_channels": RWSE_DIM,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "single_conformer": True,
        "dft_coordinates_used": False,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Geometry cache contract changed for {key}")
    if float(manifest.get("valid_geometry_fraction", 0.0)) < MIN_GEOMETRY_VALID_FRACTION:
        raise RuntimeError("Geometry cache valid fraction is below the frozen gate")
    if expected_sha256 and manifest.get("aggregate_sha256") != expected_sha256:
        raise RuntimeError("Geometry cache aggregate SHA changed")
    if sha256_file(cache_root / "canonical_validity.json") != PARENT_CANONICAL_VALIDITY_SHA256:
        raise RuntimeError("Canonical validity evidence changed")
    if sha256_file(cache_root / "geometry_failures.json") != manifest.get(
        "geometry_failures_sha256"
    ):
        raise RuntimeError("Geometry failure evidence changed")
    if _aggregate_shards(manifest["shards"]) != manifest["aggregate_sha256"]:
        raise RuntimeError("Stored geometry aggregate changed")
    counts = {"train": 0, "validation": 0}
    for shard in manifest["shards"]:
        path = cache_root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Geometry shard changed: {path.name}")
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        if len(graphs) != shard["graph_count"]:
            raise RuntimeError(f"Geometry shard count changed: {path.name}")
        counts[shard["role"]] += len(graphs)
        for graph in graphs:
            if (
                tuple(graph.x.shape[1:]) != (ATOM_FEATURE_CHANNELS,)
                or tuple(graph.edge_attr.shape[1:]) != (BOND_FEATURE_CHANNELS,)
                or tuple(graph.random_walk_pe.shape[1:]) != (RWSE_DIM,)
                or tuple(graph.pos.shape) != (graph.num_nodes, 3)
                or tuple(graph.edge_distance.shape) != (graph.edge_index.shape[1], 1)
                or tuple(graph.wedge_angle_cos.shape)
                != (graph.wedge_edge_ids.shape[0], 1)
            ):
                raise RuntimeError(f"Graph tensor contract changed: {path.name}")
            for value in (
                graph.random_walk_pe,
                graph.pos,
                graph.edge_distance,
                graph.wedge_angle_cos,
            ):
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"Non-finite cache tensor: {path.name}")
    if counts != required["roles"]:
        raise RuntimeError(f"Geometry role counts changed: {counts}")
    return manifest


def load_cache(cache_root: Path, expected_sha256: str) -> tuple[dict, dict]:
    import torch

    manifest = verify_cache(cache_root, expected_sha256)
    acceptance_path = cache_root / "acceptance.json"
    if not acceptance_path.is_file():
        raise RuntimeError("Independent geometry cache acceptance is missing")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    for key, value in {
        "format": "molgap-qm9-adaptive-denoising-cache-acceptance-v1",
        "accepted": True,
        "source_commit": manifest["source_commit"],
        "cache_aggregate_sha256": expected_sha256,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }.items():
        if acceptance.get(key) != value:
            raise RuntimeError(f"Cache acceptance changed for {key}")
    roles = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        roles[shard["role"]].extend(
            torch.load(
                cache_root / shard["file"], map_location="cpu", weights_only=False
            )
        )
    return roles, manifest


def make_encoder():
    from .pcqm_gap_architecture import OGBGeometrySparseTriangleEdgeStateGPSWrapper

    return OGBGeometrySparseTriangleEdgeStateGPSWrapper(
        in_channels=ATOM_FEATURE_CHANNELS,
        edge_dim=BOND_FEATURE_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        num_heads=4,
        dropout=0.05,
        n_targets=1,
        pooling="mean",
        rwse_dim=RWSE_DIM,
        edge_state_channels=EDGE_STATE_CHANNELS,
        wedge_channels=16,
        geometry_mode="distance_angle",
        geometry_basis_channels=16,
    )


def make_loader(graphs, *, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    validate_screen_arm(
        physical_batch_per_device=BATCH_SIZE,
        device_count=1,
        gradient_accumulation_steps=1,
    )
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def forward_gap(model, batch, *, edge_distance=None, wedge_angle_cos=None):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance if edge_distance is None else edge_distance,
        batch.wedge_angle_cos if wedge_angle_cos is None else wedge_angle_cos,
        batch.geometry_valid,
    ).view(-1)


def _assert_fp32_contract(model, batch) -> None:
    import torch

    if torch.is_autocast_enabled():
        raise RuntimeError("Autocast is forbidden by the FP32 screen contract")
    floating_inputs = (
        batch.random_walk_pe,
        batch.pos,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    )
    if any(value.dtype != torch.float32 for value in floating_inputs):
        raise RuntimeError("A floating input is not FP32")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise RuntimeError("A model parameter is not FP32")


def geometry_from_positions(pos, edge_index, wedge_edge_ids):
    import torch

    source, target = edge_index
    vectors = pos[source] - pos[target]
    distance = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = pos.new_zeros((wedge_edge_ids.shape[0], 1))
    if wedge_edge_ids.shape[0]:
        first, second = wedge_edge_ids.unbind(dim=1)
        left = edge_index[0, first]
        center = edge_index[1, first]
        right = edge_index[1, second]
        left_vector = pos[left] - pos[center]
        right_vector = pos[right] - pos[center]
        denominator = (
            left_vector.norm(dim=-1) * right_vector.norm(dim=-1)
        ).clamp_min(1e-8)
        angle[:, 0] = (
            (left_vector * right_vector).sum(dim=-1) / denominator
        ).clamp(-1.0, 1.0)
    if not torch.isfinite(distance).all() or not torch.isfinite(angle).all():
        raise RuntimeError("Corrupted geometry became non-finite")
    return distance, angle


def clean_local_environment(model, batch):
    """Return invariant atom-specific clean 2D environments."""
    h = model._embed_nodes(batch.x)
    h = h + model.rwse_encoder(batch.random_walk_pe.float())
    source, target = batch.edge_index
    neighbor = h.new_zeros(h.shape)
    count = h.new_zeros((h.shape[0], 1))
    neighbor.index_add_(0, target, h[source])
    count.index_add_(0, target, h.new_ones((source.shape[0], 1)))
    return np_to_torch_cat((h, neighbor / count.clamp_min(1.0)), dim=-1)


def np_to_torch_cat(values, dim: int):
    import torch

    return torch.cat(values, dim=dim)


class InvariantNoiseGenerator:
    """Factory for the pretrain-only atom-specific isotropic noise model."""

    @staticmethod
    def make():
        import torch
        import torch.nn as nn

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.LayerNorm(2 * HIDDEN_CHANNELS),
                    nn.Linear(2 * HIDDEN_CHANNELS, 96),
                    nn.SiLU(),
                    nn.Linear(96, 1),
                )
                nn.init.zeros_(self.network[-1].weight)
                nn.init.zeros_(self.network[-1].bias)

            def forward(self, environment):
                log_prior = math.log(DENOISING_PRIOR_SIGMA)
                log_sigma = log_prior + ADAPTIVE_LOG_SCALE_RADIUS * torch.tanh(
                    self.network(environment)
                )
                return log_sigma

        return Generator()


class EquivariantDenoisingHead:
    """Factory for a bond-direction vector readout from invariant node states."""

    @staticmethod
    def make():
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_coeff = nn.Sequential(
                    nn.LayerNorm(2 * HIDDEN_CHANNELS + 1),
                    nn.Linear(2 * HIDDEN_CHANNELS + 1, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                )

            def forward(self, node_state, edge_index, corrupted_pos):
                source, target = edge_index
                vectors = corrupted_pos[source] - corrupted_pos[target]
                distances = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                unit_vectors = vectors / distances
                coefficients = self.edge_coeff(
                    np_to_torch_cat(
                        (node_state[source], node_state[target], distances), dim=-1
                    )
                )
                messages = coefficients * unit_vectors
                prediction = node_state.new_zeros((node_state.shape[0], 3))
                counts = node_state.new_zeros((node_state.shape[0], 1))
                prediction.index_add_(0, source, messages)
                counts.index_add_(0, source, counts.new_ones((source.shape[0], 1)))
                return prediction / counts.clamp_min(1.0)

        return Head()


class _NodeStateCapture:
    def __init__(self, model):
        self.node_state = None
        self.handle = model.convs[-1].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.node_state = output

    def close(self) -> None:
        self.handle.remove()


def _target_stats(graphs):
    import torch

    values = torch.stack([graph.y.view(()) for graph in graphs])
    return values.mean(), values.std().clamp_min(1e-6)


def _rng_state(loader, corruption_generator=None) -> dict:
    import torch

    result = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": loader.generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    if corruption_generator is not None:
        result["corruption"] = corruption_generator.get_state()
    return result


def _restore_rng(state: dict, loader, corruption_generator=None) -> None:
    import torch

    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    loader.generator.set_state(state["loader"])
    torch.cuda.set_rng_state_all(state["cuda"])
    if corruption_generator is not None:
        corruption_generator.set_state(state["corruption"])


def _evaluate(model, loader, mean, std, device):
    import torch

    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = forward_gap(model, batch) * std + mean
            targets.append(batch.y.view(-1).cpu())
            predictions.append(prediction.cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    return float((prediction - target).abs().mean()), target, prediction


def train_gap(
    model,
    roles,
    output_dir: Path,
    *,
    epochs: int,
    arm: str,
    source_commit: str,
    cache_sha256: str,
) -> tuple[object, dict]:
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    mean, std = _target_stats(roles["train"])
    mean, std = mean.to(device), std.to(device)
    train_loader = make_loader(roles["train"], shuffle=True, seed=SEED)
    validation_loader = make_loader(
        roles["validation"], shuffle=False, seed=SEED
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE
    )
    trace = []
    best = float("inf")
    best_epoch = -1
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        expected = {
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "phase": "direct_gap",
            "arm": arm,
            "max_epochs": epochs,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Gap checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        best = float(checkpoint["best"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], train_loader)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            _assert_fp32_contract(model, batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_gap(model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float(
                (prediction.detach() * std + mean - batch.y.view(-1)).abs().sum()
            )
            rows += int(target.numel())
        validation_mae, target_eV, prediction_eV = _evaluate(
            model, validation_loader, mean, std, device
        )
        improved = validation_mae < best
        if improved:
            best = validation_mae
            best_epoch = epoch
            atomic_torch_save(output_dir / "best_model.pt", model.state_dict())
            atomic_torch_save(
                output_dir / "best_validation_payload.pt",
                {"target_eV": target_eV, "prediction_eV": prediction_eV},
            )
        seconds = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": absolute / rows,
            "validation_gap_mae_eV": validation_mae,
            "seconds": seconds,
            "throughput_graphs_per_s": TRAIN_ROWS / seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
        }
        trace.append(row)
        scheduler.step()
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(train_loader),
                "trace": trace,
                "best": best,
                "best_epoch": best_epoch,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "phase": "direct_gap",
                "arm": arm,
                "max_epochs": epochs,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "precision_verified": True,
                "autocast_enabled": False,
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"{arm} gap ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"val={validation_mae:.6f}eV {seconds:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )

    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    )
    payload_path = output_dir / "best_validation_payload.pt"
    return model, {
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "throughput_graphs_per_s": float(
            np.mean([row["throughput_graphs_per_s"] for row in trace])
        ),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "model_sha256": sha256_file(output_dir / "best_model.pt"),
        "payload_sha256": sha256_file(payload_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def pretrain_denoising(
    model,
    roles,
    output_dir: Path,
    *,
    adaptive: bool,
    arm: str,
    source_commit: str,
    cache_sha256: str,
) -> tuple[object, dict]:
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    capture = _NodeStateCapture(model)
    set_seed(SEED + 1)
    head = EquivariantDenoisingHead.make().to(device)
    denoising_head_initial_sha256 = state_sha256(head)
    adaptive_noise_generator = None
    if adaptive:
        set_seed(SEED + 2)
        adaptive_noise_generator = InvariantNoiseGenerator.make().to(device)
    # Auxiliary construction must not shift dropout/random streams between the
    # fixed and adaptive arms; only the learned sigma mechanism may differ.
    set_seed(SEED + 100)
    parameters = list(model.parameters()) + list(head.parameters())
    if adaptive_noise_generator is not None:
        parameters += list(adaptive_noise_generator.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    loader = make_loader(roles["train"], shuffle=True, seed=SEED)
    corruption_generator = torch.Generator(device=device).manual_seed(SEED + 3)
    trace = []
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        expected = {
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "phase": "adaptive_denoising" if adaptive else "fixed_denoising",
            "arm": arm,
            "max_epochs": PRETRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Denoising checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])
        if adaptive_noise_generator is not None:
            adaptive_noise_generator.load_state_dict(checkpoint["generator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], loader, corruption_generator)

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, PRETRAIN_EPOCHS):
        model.train()
        head.train()
        totals = {"pretrain_loss": 0.0, "denoise_loss": 0.0, "kl_loss": 0.0}
        sigma_values = []
        graphs = 0
        batches = 0
        started = time.perf_counter()
        for batch in loader:
            batch = batch.to(device)
            _assert_fp32_contract(model, batch)
            optimizer.zero_grad(set_to_none=True)
            if adaptive_noise_generator is None:
                sigma = batch.pos.new_full((batch.num_nodes, 1), DENOISING_SIGMA)
            else:
                environment = clean_local_environment(model, batch)
                log_sigma = adaptive_noise_generator(environment)
                sigma = log_sigma.exp()
            # reparameterized atom-local isotropic corruption.
            epsilon = torch.randn(
                batch.pos.shape,
                generator=corruption_generator,
                device=device,
                dtype=batch.pos.dtype,
            )
            graph_valid = batch.geometry_valid.view(-1).float()
            atom_valid = graph_valid[batch.batch].view(-1, 1)
            corrupted_pos = batch.pos + sigma * epsilon * atom_valid
            edge_distance, wedge_angle_cos = geometry_from_positions(
                corrupted_pos, batch.edge_index, batch.wedge_edge_ids
            )
            forward_gap(
                model,
                batch,
                edge_distance=edge_distance,
                wedge_angle_cos=wedge_angle_cos,
            )
            if capture.node_state is None:
                raise RuntimeError("Final node state hook did not fire")
            predicted_noise = head(
                capture.node_state, batch.edge_index, corrupted_pos
            )
            valid_count = atom_valid.sum().clamp_min(1.0)
            denoise_loss = (
                ((predicted_noise - epsilon) ** 2) * atom_valid
            ).sum() / (3.0 * valid_count)
            if adaptive_noise_generator is None:
                kl_loss = denoise_loss.new_zeros(())
            else:
                ratio = (sigma / DENOISING_PRIOR_SIGMA).pow(2)
                kl_loss = (0.5 * (ratio - 1.0 - ratio.log()) * atom_valid).sum()
                kl_loss = kl_loss / valid_count
            loss = denoise_loss + KL_WEIGHT * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["pretrain_loss"] += float(loss.detach())
            totals["denoise_loss"] += float(denoise_loss.detach())
            totals["kl_loss"] += float(kl_loss.detach())
            sigma_values.append(sigma.detach().cpu())
            graphs += int(batch.num_graphs)
            batches += 1
        scheduler.step()
        seconds = time.perf_counter() - started
        sigmas = torch.cat(sigma_values)
        row = {
            "epoch": epoch,
            **{key: value / batches for key, value in totals.items()},
            "sigma_mean": float(sigmas.mean()),
            "sigma_std": float(sigmas.std()),
            "seconds": seconds,
            "throughput_graphs_per_s": graphs / seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "head": head.state_dict(),
                "generator": (
                    adaptive_noise_generator.state_dict()
                    if adaptive_noise_generator is not None
                    else None
                ),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(loader, corruption_generator),
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "phase": "adaptive_denoising" if adaptive else "fixed_denoising",
                "arm": arm,
                "max_epochs": PRETRAIN_EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "precision_verified": True,
                "autocast_enabled": False,
                "rng_contract": {
                    "model_seed": SEED,
                    "auxiliary_head_seed": SEED + 1,
                    "adaptive_generator_seed": SEED + 2,
                    "corruption_seed": SEED + 3,
                    "training_stream_seed": SEED + 100,
                    "loader_seed": SEED,
                },
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"{arm} pretrain ep{epoch:02d} loss={row['pretrain_loss']:.6f} "
            f"sigma={row['sigma_mean']:.5f}+/-{row['sigma_std']:.5f} "
            f"{seconds:.1f}s",
            flush=True,
        )
    capture.close()
    training_parameter_count = sum(parameter.numel() for parameter in parameters)
    del head, adaptive_noise_generator
    return model, {
        "pretrain_only": True,
        "adaptive": adaptive,
        "epochs_completed": len(trace),
        "pretrain_loss": trace[-1]["pretrain_loss"],
        "denoise_loss": trace[-1]["denoise_loss"],
        "kl_loss": trace[-1]["kl_loss"],
        "sigma_mean": trace[-1]["sigma_mean"],
        "sigma_std": trace[-1]["sigma_std"],
        "denoising_head_initial_sha256": denoising_head_initial_sha256,
        "training_parameter_count": training_parameter_count,
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "throughput_graphs_per_s": float(
            np.mean([row["throughput_graphs_per_s"] for row in trace])
        ),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def arm_contract(arm: str, *, accelerator: str) -> dict:
    return {
        "arm": arm,
        "task_id": TASK_ID,
        "platform_id": PLATFORM_ID,
        "accelerator": accelerator,
        "data_role_fingerprint": SPLIT_FINGERPRINT,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "40-encoder-passes-cosine-per-objective-eta1e-6",
        "sample_exposure": "qm9-train30000-encoder40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "direct_gap": True,
    }


def run_worker(
    arm: str,
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
) -> dict:
    import torch

    if arm not in {"scratch40", "fixed10_gap30", "adaptive10_gap30"}:
        raise ValueError(f"Unknown arm: {arm}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker requires exactly one visible GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    roles, manifest = load_cache(cache_root, cache_sha256)
    output_dir = output_root / arm
    set_seed(SEED)
    model = make_encoder()
    initial_model_sha256 = state_sha256(model)
    inference_parameter_count = sum(p.numel() for p in model.parameters())
    pretrain = None
    if arm != "scratch40":
        model, pretrain = pretrain_denoising(
            model,
            roles,
            output_dir / "pretrain",
            adaptive=arm == "adaptive10_gap30",
            arm=arm,
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        )
        gap_epochs = GAP_EPOCHS
    else:
        gap_epochs = SCRATCH_EPOCHS
    model, training = train_gap(
        model,
        roles,
        output_dir / "gap",
        epochs=gap_epochs,
        arm=arm,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
    )
    result = {
        "arm": arm,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "initial_model_sha256": initial_model_sha256,
        "inference_parameter_count": inference_parameter_count,
        "parameter_count": inference_parameter_count,
        "pretraining": pretrain,
        "training": training,
        "contract": arm_contract(arm, accelerator=torch.cuda.get_device_name(0)),
        "encoder_epochs_completed": (
            training["epochs_completed"]
            + (pretrain["epochs_completed"] if pretrain is not None else 0)
        ),
        "model_inference_executed": True,
        "precision_verified": True,
        "autocast_enabled": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / f"{arm}_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    names = ("scratch40", "fixed10_gap30", "adaptive10_gap30")
    results = {
        name: json.loads(
            (output_root / f"{name}_worker.json").read_text(encoding="utf-8")
        )
        for name in names
    }
    initial_hashes = {result["initial_model_sha256"] for result in results.values()}
    if len(initial_hashes) != 1:
        raise RuntimeError("Downstream encoder initialization differs across arms")
    if any(result["encoder_epochs_completed"] != TOTAL_ENCODER_EPOCHS for result in results.values()):
        raise RuntimeError("An arm did not complete exactly 40 encoder passes")
    fixed_head = results["fixed10_gap30"]["pretraining"][
        "denoising_head_initial_sha256"
    ]
    adaptive_head = results["adaptive10_gap30"]["pretraining"][
        "denoising_head_initial_sha256"
    ]
    if fixed_head != adaptive_head:
        raise RuntimeError("Fixed/adaptive denoising head initialization differs")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in names]
    )
    values = {
        name: float(results[name]["training"]["validation_gap_mae_eV"])
        for name in names
    }
    gain_vs_scratch = values["scratch40"] - values["adaptive10_gap30"]
    gain_vs_fixed = values["fixed10_gap30"] - values["adaptive10_gap30"]
    nominated = (
        gain_vs_scratch >= MIN_GAIN_VS_SCRATCH_EV
        and gain_vs_fixed >= MIN_GAIN_VS_FIXED_EV
    )
    metrics = {
        "format": "molgap-qm9-adaptive-denoising-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "comparability": comparability,
        "results": results,
        "validation_gap_mae_eV": values,
        "adaptive_gain_vs_scratch_eV": gain_vs_scratch,
        "adaptive_gain_vs_fixed_eV": gain_vs_fixed,
        "required_gain_vs_scratch_eV": MIN_GAIN_VS_SCRATCH_EV,
        "required_gain_vs_fixed_eV": MIN_GAIN_VS_FIXED_EV,
        "pcqm100k_transfer_nominated": nominated,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "metrics.json", metrics)
    atomic_json(
        output_root / "completion_manifest.json",
        {
            **metrics,
            "artifact_sha256": {
                str(path.relative_to(output_root)): sha256_file(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "completion_manifest.json"
            },
        },
    )
    return metrics
