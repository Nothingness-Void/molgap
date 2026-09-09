"""Paired QM9 screen for a deterministic atom-level charge prior.

The candidate changes only the initial node state of the accepted OGB
EdgeState Structural GPS9. Gasteiger atom and implicit-hydrogen charges are
standardized with train-only statistics and injected through a zero-output
low-rank adapter. Both arms execute the same adapter path and start from
identical state; the sham control keeps the zero adapter frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np


TRAIN_ROWS = 30_000
VALIDATION_ROWS = 3_000
SPLIT_SEED = 42
MODEL_SEED = 42
RWSE_DIM = 16
BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 40
PATIENCE = 8
MIN_GAIN_EV = 0.001
CHARGE_CHANNELS = 2
ADAPTER_RANK = 16
CHARGE_ALGORITHM = "rdkit-gasteiger-atom-hydrogen-niter12-v1"
CACHE_FORMAT = "molgap-qm9-charge-adapter-cache-v1"
ACCEPTANCE_FORMAT = "molgap-qm9-charge-adapter-cache-acceptance-v1"
SCREEN_FORMAT = "molgap-qm9-charge-adapter-screen-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _torch_load(path: Path, map_location="cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def _source_contract(source_root: Path):
    from rdkit import rdBase

    from .qm9_data import load_qm9_records, prepare_qm9_files
    from .qm9_local_hierarchy import (
        EXPECTED_PROCESSED_SHA256,
        EXPECTED_RAW_SDF_SHA256,
        EXPECTED_RDKIT_VERSION,
        EXPECTED_SOURCE_ROWS,
    )

    records = load_qm9_records(source_root)
    files = prepare_qm9_files(source_root)
    contract = {
        "source_rows": len(records),
        "rdkit_version": rdBase.rdkitVersion,
        "processed_sha256": sha256_file(files["processed"]),
        "raw_sdf_sha256": sha256_file(files["raw_sdf"]),
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "rdkit_version": EXPECTED_RDKIT_VERSION,
        "processed_sha256": EXPECTED_PROCESSED_SHA256,
        "raw_sdf_sha256": EXPECTED_RAW_SDF_SHA256,
    }
    if contract != expected:
        raise RuntimeError(f"Frozen QM9 source contract changed: {contract}")
    return records, files, contract


def _gasteiger_features(molecule):
    import torch
    from rdkit.Chem import rdPartialCharges

    rdPartialCharges.ComputeGasteigerCharges(
        molecule, nIter=12, throwOnParamFailure=True
    )
    values = []
    for atom in molecule.GetAtoms():
        values.append(
            [
                float(atom.GetProp("_GasteigerCharge")),
                float(atom.GetProp("_GasteigerHCharge")),
            ]
        )
    features = torch.tensor(values, dtype=torch.float32)
    if features.shape != (molecule.GetNumAtoms(), CHARGE_CHANNELS):
        raise RuntimeError("Gasteiger feature shape changed")
    if not torch.isfinite(features).all():
        raise RuntimeError("Gasteiger calculation produced non-finite values")
    return features


def _make_graph(record: dict, source_idx: int, supplier):
    from .qm9_local_hierarchy import _make_graph as make_base_graph
    from .qm9_local_hierarchy import _record_molecule

    molecule, _ = _record_molecule(record, supplier)
    graph = make_base_graph(record, source_idx, supplier)
    graph.gasteiger_features = _gasteiger_features(molecule)
    if graph.gasteiger_features.shape[0] != graph.x.shape[0]:
        raise RuntimeError("Gasteiger features are not atom-aligned")
    return graph


def _validate_graph(graph, expected_source_idx: int | None = None) -> None:
    import torch

    if expected_source_idx is not None:
        observed = int(graph.row_id.view(-1)[0])
        if observed != int(expected_source_idx):
            raise RuntimeError(
                f"row identity changed: {observed} != {expected_source_idx}"
            )
    node_count = int(graph.x.shape[0])
    if graph.x.ndim != 2 or graph.x.shape[1] != 9:
        raise RuntimeError("OGB atom representation changed")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
        raise RuntimeError("OGB bond representation changed")
    if graph.random_walk_pe.shape != (node_count, RWSE_DIM):
        raise RuntimeError("RWSE is not atom-aligned")
    if graph.gasteiger_features.shape != (node_count, CHARGE_CHANNELS):
        raise RuntimeError("Gasteiger features are not atom-aligned")
    for value in (
        graph.x,
        graph.edge_attr,
        graph.y,
        graph.random_walk_pe,
        graph.gasteiger_features,
    ):
        if not torch.isfinite(value.float()).all():
            raise RuntimeError("cache contains non-finite values")


def _charge_stats(graphs) -> dict:
    import torch

    values = torch.cat([graph.gasteiger_features.double() for graph in graphs])
    return {
        "atom_rows": int(values.shape[0]),
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).clamp_min(1e-8).tolist(),
        "minimum": values.min(dim=0).values.tolist(),
        "maximum": values.max(dim=0).values.tolist(),
    }


def build_cache(
    output_root: Path,
    *,
    source_root: Path,
    source_commit: str,
    shard_size: int = 2_000,
):
    """Build an atomic, resumable train/validation-only charge cache."""
    import torch
    from rdkit import Chem, RDLogger

    from .qm9_data import fixed_split_from_pool
    from .qm9_local_hierarchy import _canonical_valid_pool

    if len(source_commit) != 40:
        raise ValueError("A full source commit is required")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final cache manifest exists")

    records, files, source = _source_contract(source_root)
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(
        str(files["raw_sdf"]), removeHs=False, sanitize=False
    )
    validity_path = output_root / "canonical_validity.json"
    if validity_path.is_file():
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        if (
            validity.get("source") != source
            or validity.get("source_commit") != source_commit
            or validity.get("charge_algorithm") != CHARGE_ALGORITHM
        ):
            raise RuntimeError("Canonical validity contract changed")
        valid_pool = np.asarray(validity["valid_source_indices"], dtype=np.int64)
        train_indices = np.asarray(
            validity["selected_train_indices"], dtype=np.int64
        )
        validation_indices = np.asarray(
            validity["selected_validation_indices"], dtype=np.int64
        )
        invalid = validity["invalid_records"]
    else:
        valid_pool, invalid = _canonical_valid_pool(records, supplier)
        split = fixed_split_from_pool(
            valid_pool, TRAIN_ROWS, VALIDATION_ROWS, 0, SPLIT_SEED
        )
        train_indices, validation_indices = split.train, split.validation
        validity = {
            "format": "molgap-qm9-charge-canonical-validity-v1",
            "source": source,
            "source_commit": source_commit,
            "charge_algorithm": CHARGE_ALGORITHM,
            "valid_source_indices": valid_pool.tolist(),
            "invalid_records": invalid,
            "selected_train_indices": train_indices.tolist(),
            "selected_validation_indices": validation_indices.tolist(),
            "valid_source_indices_sha256": _indices_sha256(valid_pool),
            "selected_train_indices_sha256": _indices_sha256(train_indices),
            "selected_validation_indices_sha256": _indices_sha256(
                validation_indices
            ),
            "held_out_indices_materialized": False,
            "test_role_materialized": False,
            "official_pcqm_roles_read": False,
        }
        atomic_json(validity_path, validity)

    roles = {"train": train_indices, "validation": validation_indices}
    progress_path = output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"completed_shards": []}
    )
    previous = {item["file"]: item for item in progress["completed_shards"]}
    shards = []
    train_graphs = []
    for role, indices in roles.items():
        for part_number, start in enumerate(range(0, len(indices), shard_size)):
            expected = np.asarray(
                indices[start : start + shard_size], dtype=np.int64
            )
            filename = f"{role}_part_{part_number:03d}.pt"
            path = output_root / filename
            if path.is_file():
                graphs = _torch_load(path)
            else:
                graphs = [
                    _make_graph(records[int(index)], int(index), supplier)
                    for index in expected
                ]
                atomic_torch_save(path, graphs)
            if len(graphs) != len(expected):
                raise RuntimeError(f"Shard count changed: {filename}")
            for graph, index in zip(graphs, expected):
                _validate_graph(graph, int(index))
            if role == "train":
                train_graphs.extend(graphs)
            shard = {
                "role": role,
                "file": filename,
                "graph_count": len(graphs),
                "sha256": sha256_file(path),
            }
            if filename in previous and previous[filename] != shard:
                raise RuntimeError(f"Resumed shard changed: {filename}")
            shards.append(shard)
            atomic_json(
                progress_path,
                {
                    "format": f"{CACHE_FORMAT}-progress",
                    "source": source,
                    "source_commit": source_commit,
                    "charge_algorithm": CHARGE_ALGORITHM,
                    "completed_shards": shards,
                    "held_out_indices_materialized": False,
                    "test_role_materialized": False,
                    "official_pcqm_roles_read": False,
                },
            )
            print(f"{role} shard {part_number + 1}: {len(graphs)} graphs", flush=True)

    aggregate = hashlib.sha256()
    for shard in shards:
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    manifest = {
        "format": CACHE_FORMAT,
        "complete": True,
        "source": source,
        "source_commit": source_commit,
        "charge_algorithm": CHARGE_ALGORITHM,
        "split_seed": SPLIT_SEED,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "held_out_indices_materialized": False,
        "test_role_materialized": False,
        "official_pcqm_roles_read": False,
        "canonical_valid_pool_count": int(len(valid_pool)),
        "canonical_invalid_count": len(invalid),
        "canonical_validity_sha256": sha256_file(validity_path),
        "canonical_valid_pool_sha256": _indices_sha256(valid_pool),
        "train_source_indices_sha256": _indices_sha256(train_indices),
        "validation_source_indices_sha256": _indices_sha256(validation_indices),
        "split_fingerprint": hashlib.sha256(
            np.concatenate((train_indices, validation_indices))
            .astype(np.int64)
            .tobytes()
        ).hexdigest()[:16],
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": RWSE_DIM,
        "charge_channels": CHARGE_CHANNELS,
        "train_charge_statistics": _charge_stats(train_graphs),
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "gpu_used": False,
        "model_inference_executed": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def accept_cache(cache_root: Path, *, expected_source_commit: str | None = None):
    manifest_path = cache_root / "manifest.json"
    validity_path = cache_root / "canonical_validity.json"
    if not manifest_path.is_file() or not validity_path.is_file():
        raise FileNotFoundError("Cache manifest or canonical validity is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "format": CACHE_FORMAT,
        "complete": True,
        "charge_algorithm": CHARGE_ALGORITHM,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "held_out_indices_materialized": False,
        "test_role_materialized": False,
        "official_pcqm_roles_read": False,
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": RWSE_DIM,
        "charge_channels": CHARGE_CHANNELS,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Cache contract changed for {key}")
    if expected_source_commit and manifest["source_commit"] != expected_source_commit:
        raise RuntimeError("Cache source commit does not match submitted code")
    if sha256_file(validity_path) != manifest["canonical_validity_sha256"]:
        raise RuntimeError("Canonical validity hash changed")
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    pool = np.asarray(validity["valid_source_indices"], dtype=np.int64)
    train = np.asarray(validity["selected_train_indices"], dtype=np.int64)
    validation = np.asarray(
        validity["selected_validation_indices"], dtype=np.int64
    )
    if (
        validity.get("format") != "molgap-qm9-charge-canonical-validity-v1"
        or validity.get("source") != manifest.get("source")
        or validity.get("source_commit") != manifest.get("source_commit")
        or validity.get("charge_algorithm") != CHARGE_ALGORITHM
        or validity.get("held_out_indices_materialized") is not False
        or validity.get("test_role_materialized") is not False
        or validity.get("official_pcqm_roles_read") is not False
        or len(train) != TRAIN_ROWS
        or len(validation) != VALIDATION_ROWS
        or np.intersect1d(train, validation).size
        or not np.isin(train, pool).all()
        or not np.isin(validation, pool).all()
        or _indices_sha256(pool) != manifest["canonical_valid_pool_sha256"]
        or _indices_sha256(train) != manifest["train_source_indices_sha256"]
        or _indices_sha256(validation)
        != manifest["validation_source_indices_sha256"]
    ):
        raise RuntimeError("Canonical-valid split contract changed")

    aggregate = hashlib.sha256()
    observed = {"train": 0, "validation": 0}
    expected_indices = {"train": train, "validation": validation}
    role_offsets = {"train": 0, "validation": 0}
    train_graphs = []
    for shard in manifest["shards"]:
        path = cache_root / shard["file"]
        observed_sha = sha256_file(path)
        if observed_sha != shard["sha256"]:
            raise RuntimeError(f"Cache shard hash changed: {path.name}")
        graphs = _torch_load(path)
        if len(graphs) != shard["graph_count"]:
            raise RuntimeError(f"Cache shard count changed: {path.name}")
        role = shard["role"]
        start = role_offsets[role]
        expected = expected_indices[role][start : start + len(graphs)]
        for graph, source_idx in zip(graphs, expected):
            _validate_graph(graph, int(source_idx))
        observed[role] += len(graphs)
        role_offsets[role] += len(graphs)
        if role == "train":
            train_graphs.extend(graphs)
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{observed_sha}\n".encode(
                "ascii"
            )
        )
    if observed != required["roles"]:
        raise RuntimeError(f"Cache role counts changed: {observed}")
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Cache aggregate hash changed")
    recomputed = _charge_stats(train_graphs)
    expected_stats = manifest["train_charge_statistics"]
    if recomputed["atom_rows"] != expected_stats["atom_rows"]:
        raise RuntimeError("Train charge atom count changed")
    for key in ("mean", "std", "minimum", "maximum"):
        if not np.allclose(recomputed[key], expected_stats[key], atol=1e-12):
            raise RuntimeError(f"Train charge statistic changed: {key}")
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "accepted": True,
        "source_commit": manifest["source_commit"],
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "canonical_validity_sha256": manifest["canonical_validity_sha256"],
        "role_counts": observed,
        "charge_algorithm": CHARGE_ALGORITHM,
        "charge_statistics_verified": True,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(cache_root / "acceptance.json", acceptance)
    return acceptance


def load_cache(cache_root: Path, expected_sha256: str | None = None):
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads(
        (cache_root / "acceptance.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("format") != CACHE_FORMAT
        or manifest.get("complete") is not True
        or acceptance.get("format") != ACCEPTANCE_FORMAT
        or acceptance.get("accepted") is not True
        or acceptance.get("cache_aggregate_sha256")
        != manifest.get("aggregate_sha256")
    ):
        raise RuntimeError("Charge cache has not passed acceptance")
    if expected_sha256 and manifest["aggregate_sha256"] != expected_sha256:
        raise RuntimeError("Cache aggregate SHA changed")
    roles = {"train": [], "validation": []}
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = cache_root / shard["file"]
        observed_sha = sha256_file(path)
        if observed_sha != shard["sha256"]:
            raise RuntimeError(f"Cache shard changed: {path.name}")
        roles[shard["role"]].extend(_torch_load(path))
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{observed_sha}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Cache aggregate hash changed")
    if {key: len(value) for key, value in roles.items()} != manifest["roles"]:
        raise RuntimeError("Loaded role counts changed")
    return roles, manifest


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_control_encoder():
    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    return OGBEdgeStateStructuralGPSWrapper(
        in_channels=9,
        edge_dim=3,
        hidden_channels=192,
        num_layers=9,
        num_heads=4,
        dropout=0.05,
        n_targets=1,
        pooling="mean",
        rwse_dim=RWSE_DIM,
        edge_state_channels=64,
    )


def make_charge_encoder(charge_mean, charge_std):
    import torch
    import torch.nn as nn

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    class OGBChargeAdapterStructuralGPSWrapper(
        OGBEdgeStateStructuralGPSWrapper
    ):
        def __init__(self):
            super().__init__(
                in_channels=9,
                edge_dim=3,
                hidden_channels=192,
                num_layers=9,
                num_heads=4,
                dropout=0.05,
                n_targets=1,
                pooling="mean",
                rwse_dim=RWSE_DIM,
                edge_state_channels=64,
            )
            self.register_buffer(
                "charge_mean",
                torch.as_tensor(charge_mean, dtype=torch.float32).view(1, -1),
            )
            self.register_buffer(
                "charge_std",
                torch.as_tensor(charge_std, dtype=torch.float32).view(1, -1),
            )
            self.charge_adapter = nn.Sequential(
                nn.Linear(CHARGE_CHANNELS, ADAPTER_RANK, bias=True),
                nn.SiLU(),
                nn.Linear(ADAPTER_RANK, 192, bias=False),
            )
            nn.init.zeros_(self.charge_adapter[-1].weight)

        def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            gasteiger_features,
        ):
            return self.head(
                self.encode(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    random_walk_pe,
                    gasteiger_features,
                )
            )

        def encode(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            random_walk_pe,
            gasteiger_features,
        ):
            if random_walk_pe.shape != (x.shape[0], RWSE_DIM):
                raise ValueError("RWSE shape changed")
            if gasteiger_features.shape != (x.shape[0], CHARGE_CHANNELS):
                raise ValueError("Gasteiger feature shape changed")
            if not torch.isfinite(gasteiger_features).all():
                raise ValueError("Gasteiger features contain non-finite values")
            normalized = (
                gasteiger_features.float() - self.charge_mean
            ) / self.charge_std
            h = self._embed_nodes(x)
            h = h + self.rwse_encoder(random_walk_pe.float())
            h = h + self.charge_adapter(normalized)
            edge_state = self._embed_edges(edge_attr)
            graph_state = self._initialize_graph_state(h, batch)
            for edge_update, conv in zip(self.edge_updates, self.convs):
                edge_state = edge_update(h, edge_index, edge_state)
                h = self._condition_nodes_from_edges(h, edge_index, edge_state)
                h = self._condition_nodes_from_graph(h, batch, graph_state)
                h = conv(h, edge_index, batch, edge_attr=edge_state)
                graph_state = self._update_graph_state(h, batch, graph_state)
            return self._pool(h, batch)

    return OGBChargeAdapterStructuralGPSWrapper()


def freeze_charge_adapter(model):
    """Turn the charge path into an identical-compute frozen-zero sham."""
    for parameter in model.charge_adapter.parameters():
        parameter.requires_grad_(False)
    return model


def forward_encoder(model, batch, *, candidate: bool):
    arguments = (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    if candidate:
        return model(*arguments, batch.gasteiger_features).view(-1)
    return model(*arguments).view(-1)


def _make_loader(graphs, *, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    from .screen_policy import validate_screen_arm

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


def _arm_contract(arm: str, *, accelerator: str, split_fingerprint: str) -> dict:
    return {
        "arm": arm,
        "task_id": "qm9-charge-adapter-s42-v2",
        "platform_id": "scnet-kunshan",
        "accelerator": accelerator,
        "data_role_fingerprint": split_fingerprint,
        "seed": MODEL_SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine40-eta1e-6-patience8",
        "sample_exposure": "qm9-train30000-gap40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def _target_stats(graphs):
    import torch

    targets = torch.stack([graph.y.view(()) for graph in graphs])
    return targets.mean(), targets.std().clamp_min(1e-6)


def _state_sha256(model, *, shared_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if shared_only and (
            name.startswith("charge_adapter.")
            or name in {"charge_mean", "charge_std"}
        ):
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shared_initialization_exact(control, candidate) -> bool:
    candidate_state = candidate.state_dict()
    return all(
        name in candidate_state and value.equal(candidate_state[name])
        for name, value in control.state_dict().items()
    )


def _rng_state(loader):
    import torch

    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": loader.generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state, loader):
    import torch

    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    loader.generator.set_state(state["loader"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _evaluate(model, loader, mean, std, device, *, candidate: bool):
    import torch

    model.eval()
    targets = []
    predictions = []
    row_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = forward_encoder(model, batch, candidate=candidate) * std + mean
            targets.append(batch.y.view(-1).detach().cpu())
            predictions.append(prediction.detach().cpu())
            row_ids.append(batch.row_id.view(-1).detach().cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    return float((prediction - target).abs().mean()), {
        "row_id": torch.cat(row_ids),
        "target_eV": target,
        "prediction_eV": prediction,
    }


def train_arm(
    model,
    roles,
    output_dir: Path,
    *,
    candidate: bool,
    source_commit: str,
    cache_sha256: str,
):
    import torch
    import torch.nn.functional as functional

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    mean, std = _target_stats(roles["train"])
    mean, std = mean.to(device), std.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    train_loader = _make_loader(roles["train"], shuffle=True, seed=MODEL_SEED)
    validation_loader = _make_loader(
        roles["validation"], shuffle=False, seed=MODEL_SEED
    )
    trace = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(checkpoint_path, map_location=device)
        required = {
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "candidate": candidate,
            "seed": MODEL_SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise RuntimeError("Checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        best = float(checkpoint["best"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], train_loader)
        if stale >= PATIENCE:
            start_epoch = EPOCHS

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_encoder(model, batch, candidate=candidate)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float(
                (prediction.detach() * std + mean - batch.y.view(-1)).abs().sum()
            )
            rows += int(target.numel())
        validation_mae, payload = _evaluate(
            model,
            validation_loader,
            mean,
            std,
            device,
            candidate=candidate,
        )
        improved = validation_mae < best
        if improved:
            best = validation_mae
            best_epoch = epoch
            stale = 0
            atomic_torch_save(output_dir / "best_model.pt", model.state_dict())
            atomic_torch_save(output_dir / "best_validation_payload.pt", payload)
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": absolute / rows,
            "validation_gap_mae_eV": validation_mae,
            "seconds": time.perf_counter() - started,
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
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "candidate": candidate,
                "seed": MODEL_SEED,
                "batch_size": BATCH_SIZE,
                "max_epochs": EPOCHS,
                "best": best,
                "best_epoch": best_epoch,
                "stale": stale,
                "rng": _rng_state(train_loader),
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"{output_dir.name} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"val={validation_mae:.6f}eV {row['seconds']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    if best_epoch < 0:
        raise RuntimeError("Training produced no finite best checkpoint")
    return {
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "model_sha256": sha256_file(output_dir / "best_model.pt"),
        "payload_sha256": sha256_file(
            output_dir / "best_validation_payload.pt"
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _first_physical_batch(cache_root: Path, expected_sha256: str):
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != expected_sha256:
        raise RuntimeError("Preflight cache SHA changed")
    acceptance = json.loads(
        (cache_root / "acceptance.json").read_text(encoding="utf-8")
    )
    if acceptance.get("accepted") is not True:
        raise RuntimeError("Preflight cache was not accepted")
    first = next(shard for shard in manifest["shards"] if shard["role"] == "train")
    if sha256_file(cache_root / first["file"]) != first["sha256"]:
        raise RuntimeError("Preflight shard hash changed")
    graphs = _torch_load(cache_root / first["file"])
    loader = _make_loader(graphs, shuffle=False, seed=MODEL_SEED)
    batch = next(iter(loader))
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Preflight did not materialize physical batch 128")
    return batch, manifest


def run_preflight(
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
):
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one visible DCU")
    output_root.mkdir(parents=True, exist_ok=True)
    batch, manifest = _first_physical_batch(cache_root, cache_sha256)
    stats = manifest["train_charge_statistics"]
    set_seed(MODEL_SEED)
    control = freeze_charge_adapter(
        make_charge_encoder(stats["mean"], stats["std"])
    )
    set_seed(MODEL_SEED)
    candidate = make_charge_encoder(stats["mean"], stats["std"])
    shared_exact = _shared_initialization_exact(control, candidate)
    if not shared_exact:
        raise RuntimeError("Shared model initialization is not exact")

    device = torch.device("cuda")
    batch = batch.to(device)
    control = control.to(device).eval()
    candidate = candidate.to(device).eval()
    with torch.no_grad():
        normalized_charge = (
            batch.gasteiger_features.float() - candidate.charge_mean
        ) / candidate.charge_std
        control_adapter_output = control.charge_adapter(normalized_charge)
        candidate_adapter_output = candidate.charge_adapter(normalized_charge)
        adapter_output_exact_zero = bool(
            torch.count_nonzero(control_adapter_output) == 0
            and torch.count_nonzero(candidate_adapter_output) == 0
        )
        control_prediction = forward_encoder(control, batch, candidate=True)
        candidate_prediction = forward_encoder(candidate, batch, candidate=True)
        initial_prediction_max_abs_diff = float(
            (control_prediction - candidate_prediction).abs().max()
        )
        initial_prediction_close = bool(
            torch.allclose(
                control_prediction,
                candidate_prediction,
                rtol=0.0,
                atol=1e-6,
            )
        )
    if not adapter_output_exact_zero:
        raise RuntimeError("Sham or candidate adapter output is not exactly zero")
    reports = {}
    for name, model, is_candidate in (
        ("frozen_zero_adapter_control", control, True),
        ("charge_adapter", candidate, True),
    ):
        torch.cuda.reset_peak_memory_stats()
        model = model.to(device).train()
        model.zero_grad(set_to_none=True)
        prediction = forward_encoder(model, batch, candidate=is_candidate)
        loss = functional.l1_loss(prediction, batch.y.view(-1))
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        reports[name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "prediction_shape": list(prediction.shape),
            "loss": float(loss.detach()),
            "finite_forward_backward": finite,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "initial_state_sha256": _state_sha256(model),
            "shared_state_sha256": _state_sha256(model, shared_only=True),
        }
        if not finite:
            raise RuntimeError(f"{name} produced non-finite values")
        del model
        torch.cuda.empty_cache()
    result = {
        "format": "molgap-qm9-charge-adapter-dcu-preflight-v2",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "split_fingerprint": manifest["split_fingerprint"],
        "architecture": "ogb_edgestate_gps9_plus_zero_start_charge_adapter",
        "control": "identical_compute_frozen_zero_charge_adapter",
        "charge_algorithm": CHARGE_ALGORITHM,
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gpu": torch.cuda.get_device_name(0),
        "shared_initialization_exact": shared_exact,
        "adapter_output_exact_zero": adapter_output_exact_zero,
        "zero_start_prediction_close_diagnostic": initial_prediction_close,
        "zero_start_prediction_atol": 1e-6,
        "zero_start_prediction_max_abs_diff": initial_prediction_max_abs_diff,
        "arms": reports,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "preflight.json", result)
    return result


def run_screen(
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
    preflight_path: Path,
):
    import torch

    from .screen_policy import validate_paired_screen_contract

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible DCU")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-charge-adapter-dcu-preflight-v2",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "charge_algorithm": CHARGE_ALGORITHM,
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "shared_initialization_exact": True,
        "adapter_output_exact_zero": True,
        "control": "identical_compute_frozen_zero_charge_adapter",
        "zero_start_prediction_atol": 1e-6,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"Preflight contract changed for {key}")
    roles, manifest = load_cache(cache_root, cache_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    stats = manifest["train_charge_statistics"]
    set_seed(MODEL_SEED)
    control = freeze_charge_adapter(
        make_charge_encoder(stats["mean"], stats["std"])
    )
    set_seed(MODEL_SEED)
    candidate = make_charge_encoder(stats["mean"], stats["std"])
    if not _shared_initialization_exact(control, candidate):
        raise RuntimeError("Shared initialization changed after preflight")

    results = {}
    for name, model, is_candidate in (
        ("control", control, True),
        ("charge_adapter", candidate, True),
    ):
        set_seed(MODEL_SEED)
        results[name] = train_arm(
            model,
            roles,
            output_root / name,
            candidate=is_candidate,
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        )
        del model
        torch.cuda.empty_cache()

    control_mae = float(results["control"]["validation_gap_mae_eV"])
    candidate_mae = float(results["charge_adapter"]["validation_gap_mae_eV"])
    gain = control_mae - candidate_mae
    nominated = gain >= MIN_GAIN_EV
    contracts = [
        _arm_contract(
            name,
            accelerator=torch.cuda.get_device_name(0),
            split_fingerprint=manifest["split_fingerprint"],
        )
        for name in ("control", "charge_adapter")
    ]
    comparability = validate_paired_screen_contract(contracts)
    summary = {
        "format": SCREEN_FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "split_fingerprint": manifest["split_fingerprint"],
        "seed": MODEL_SEED,
        "gpu": torch.cuda.get_device_name(0),
        "architecture": "ogb_edgestate_gps9_plus_zero_start_charge_adapter",
        "charge_algorithm": CHARGE_ALGORITHM,
        "training_contract": {
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "physical_batch_per_device": BATCH_SIZE,
            "precision": "fp32",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "cosine-eta1e-6",
            "gradient_clip_norm": 1.0,
        },
        "comparability": comparability,
        "arm_contracts": contracts,
        "results": results,
        "candidate_gain_eV": gain,
        "required_gain_eV": MIN_GAIN_EV,
        "pcqm_transfer_nominated": nominated,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "metrics.json", summary)
    atomic_json(
        output_root / "completion_manifest.json",
        {
            **summary,
            "artifact_sha256": {
                str(path.relative_to(output_root)): sha256_file(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "completion_manifest.json"
            },
        },
    )
    return summary
