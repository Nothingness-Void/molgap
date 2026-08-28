"""Official-train-only feature-contract screen for PCQM EdgeState GPS."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .gps import (
    CategoricalEdgeStateStructuralGPSWrapper,
    CategoricalRadicalContextEdgeStateStructuralGPSWrapper,
    EdgeStateStructuralGPSWrapper,
)
from .ogb_features import (
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    atom_to_ogb_feature_vector,
    bond_to_ogb_feature_vector,
)
from .pair_gps_2d import CategoricalPairGPS2DWrapper
from .pcqm_official_edge_state import (
    OFFICIAL_ATOM_LIST,
    PackedGraphDataset,
    _iter_official_rows,
    atomic_json,
    atomic_torch,
    load_official_splits,
    set_seed,
    sha256_file,
)
from .structural_encoding import add_random_walk_pe


FEATURE_SCHEMAS = ("legacy", "ogb")
SCHEDULE_VARIANTS = ("collapse10", "warmup20")
SCREEN_BUILDER_VERSION = "pcqm-official-train-feature-screen-v1"


@dataclass(frozen=True)
class FeatureScreenConfig:
    model_family: str = "edge_state"
    precision: str = "amp"
    hidden_channels: int = 192
    num_layers: int = 9
    num_heads: int = 4
    dropout: float = 0.05
    rwse_dim: int = 16
    edge_state_channels: int = 64
    categorical_encoder: str = "sum"
    categorical_field_channels: int = 16
    graph_context: str = "none"
    radical_context_channels: int = 16
    pair_channels: int = 64
    path_steps: int = 5
    triplet_rank: int = 16
    atom_input_channels: int = 64
    bond_input_channels: int = 32
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 4.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 40
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    minimum_learning_rate: float = 1.0e-6
    patience: int = 7
    gradient_clip: float = 1.0
    hard_job_budget_s: float = 3.0 * 3600.0


def _warmup_cosine_factor(
    epoch: int,
    *,
    max_epochs: int,
    warmup_epochs: int,
    minimum_factor: float,
) -> float:
    if not 0 <= warmup_epochs < max_epochs:
        raise ValueError("warmup_epochs must be in [0, max_epochs)")
    if not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("minimum_factor must be in [0, 1]")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    cosine_epochs = max(max_epochs - warmup_epochs - 1, 1)
    progress = min(max(epoch - warmup_epochs, 0), cosine_epochs) / cosine_epochs
    return minimum_factor + (1.0 - minimum_factor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: FeatureScreenConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler == "cosine":
        if config.warmup_epochs != 0:
            raise ValueError("cosine scheduler does not accept warmup epochs")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.minimum_learning_rate,
        )
    if config.scheduler == "warmup_cosine":
        minimum_factor = config.minimum_learning_rate / config.learning_rate
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _warmup_cosine_factor(
                epoch,
                max_epochs=config.max_epochs,
                warmup_epochs=config.warmup_epochs,
                minimum_factor=minimum_factor,
            ),
        )
    raise ValueError(f"unsupported scheduler: {config.scheduler}")


def _sha256_values(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _molecule(smiles: str) -> tuple[Chem.Mol, bool]:
    molecule = Chem.MolFromSmiles(smiles)
    sanitized = molecule is not None
    if molecule is None:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            raise ValueError("unparseable_smiles")
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
    return molecule, sanitized


def _allocate_development(groups: dict[str, list[int]], rows: int) -> dict[str, int]:
    total = sum(len(values) for values in groups.values())
    exact = {name: len(values) * rows / total for name, values in groups.items()}
    allocated = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = rows - sum(allocated.values())
    order = sorted(groups, key=lambda name: (exact[name] - allocated[name], name), reverse=True)
    for name in order[:remaining]:
        allocated[name] += 1
    if sum(allocated.values()) != rows:
        raise RuntimeError("development allocation does not reconcile")
    return allocated


def prepare_feature_screen_rows(
    archive: Path,
    output_dir: Path,
    *,
    train_rows: int = 100_000,
    development_rows: int = 10_000,
    seed: int = 20260826,
    shard_rows: int = 10_000,
) -> dict:
    """Select a deterministic radical/Gap-stratified subset of official train."""
    archive, output_dir = Path(archive), Path(output_dir)
    if min(train_rows, development_rows, shard_rows) <= 0:
        raise ValueError("screen row counts must be positive")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("archive_sha256") == sha256_file(archive)
            and manifest.get("train_rows") == train_rows
            and manifest.get("development_rows") == development_rows
            and manifest.get("seed") == seed
        ):
            for item in manifest["shards"]:
                path = output_dir / item["path"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"screen row shard changed: {path}")
            return manifest
        raise RuntimeError("feature-screen row resume contract changed")

    splits = load_official_splits(archive)
    official_train = np.asarray(splits["train"], dtype=np.int64)
    selected_rows = train_rows + development_rows
    if selected_rows > len(official_train):
        raise ValueError("feature screen exceeds official training rows")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(official_train, size=selected_rows, replace=False))
    selected_set = set(selected.tolist())
    records = []
    for source_idx, smiles, gap in _iter_official_rows(archive):
        if source_idx not in selected_set:
            continue
        molecule, sanitized = _molecule(smiles)
        radical_electrons = sum(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
        gap_bin = int(np.searchsorted([2.0, 4.0, 6.0, 8.0], gap, side="right"))
        records.append({
            "source_idx": source_idx,
            "smiles": smiles,
            "gap": gap,
            "radical": int(radical_electrons > 0),
            "radical_electrons": int(radical_electrons),
            "gap_bin": gap_bin,
            "smiles_sanitized": int(sanitized),
        })
    if len(records) != selected_rows:
        raise RuntimeError(f"selected official rows are incomplete: {len(records)}")
    frame = pd.DataFrame(records).sort_values("source_idx").reset_index(drop=True)
    frame["stratum"] = frame.radical.astype(str) + ":" + frame.gap_bin.astype(str)
    groups = {
        name: values.index.to_list()
        for name, values in frame.groupby("stratum", sort=True)
    }
    allocation = _allocate_development(groups, development_rows)
    role = np.zeros(len(frame), dtype=np.int8)
    for offset, (name, indices) in enumerate(sorted(groups.items())):
        local = np.asarray(indices, dtype=np.int64)
        np.random.default_rng(seed + 10_000 + offset).shuffle(local)
        role[local[:allocation[name]]] = 1
    if int((role == 0).sum()) != train_rows or int((role == 1).sum()) != development_rows:
        raise RuntimeError("feature-screen split counts do not reconcile")
    frame["role_code"] = role
    frame = frame.drop(columns=["stratum"])

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_index, start in enumerate(range(0, len(frame), shard_rows)):
        stop = min(start + shard_rows, len(frame))
        path = output_dir / f"rows_{shard_index:04d}.csv.gz"
        temporary = path.with_name(f".{path.name}.tmp.gz")
        frame.iloc[start:stop].to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        shards.append({
            "shard_index": shard_index,
            "path": path.name,
            "rows": stop - start,
            "sha256": sha256_file(path),
        })
    manifest = {
        "format": "molgap-pcqm4mv2-official-train-feature-screen-rows-v1",
        "status": "complete",
        "archive_sha256": sha256_file(archive),
        "seed": seed,
        "train_rows": train_rows,
        "development_rows": development_rows,
        "official_valid_used": False,
        "official_test_used": False,
        "external_data_used": False,
        "selected_source_idx_sha256": _sha256_values(frame.source_idx.to_numpy()),
        "train_source_idx_sha256": _sha256_values(
            frame.loc[frame.role_code.eq(0), "source_idx"].to_numpy()
        ),
        "development_source_idx_sha256": _sha256_values(
            frame.loc[frame.role_code.eq(1), "source_idx"].to_numpy()
        ),
        "radical_counts": {
            "train": int(frame.loc[frame.role_code.eq(0), "radical"].sum()),
            "development": int(frame.loc[frame.role_code.eq(1), "radical"].sum()),
        },
        "gap_bin_counts": {
            role_name: {
                str(int(key)): int(value)
                for key, value in frame.loc[frame.role_code.eq(role_code), "gap_bin"]
                .value_counts().sort_index().items()
            }
            for role_name, role_code in (("train", 0), ("development", 1))
        },
        "shards": shards,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _legacy_features(molecule: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    atom_to_index = {value: index for index, value in enumerate(OFFICIAL_ATOM_LIST)}
    nodes = []
    for atom in molecule.GetAtoms():
        one_hot = [0.0] * len(OFFICIAL_ATOM_LIST)
        one_hot[atom_to_index[atom.GetAtomicNum()]] = 1.0
        nodes.append(one_hot + [
            atom.GetDegree() / 4.0,
            atom.GetFormalCharge() / 2.0,
            float(atom.GetIsAromatic()),
        ])
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    edges = []
    for bond in molecule.GetBonds():
        value = bond_types[bond.GetBondType()]
        row = [float(value == index) for index in range(4)]
        edges.extend((row, row))
    return (
        torch.tensor(nodes, dtype=torch.float32),
        torch.tensor(edges, dtype=torch.float32) if edges else torch.zeros((0, 4)),
    )


def _ogb_features(molecule: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = [atom_to_ogb_feature_vector(atom) for atom in molecule.GetAtoms()]
    edges = []
    for bond in molecule.GetBonds():
        row = bond_to_ogb_feature_vector(bond)
        edges.extend((row, row))
    return (
        torch.tensor(nodes, dtype=torch.long),
        torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 3), dtype=torch.long),
    )


def _graphs_from_screen_row(row) -> dict[str, Data]:
    molecule, sanitized = _molecule(str(row.smiles))
    rows, columns = [], []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows.extend((left, right))
        columns.extend((right, left))
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    shared = {
        "edge_index": edge_index,
        "y": torch.tensor([float(row.gap)], dtype=torch.float32),
        "source_idx": torch.tensor([int(row.source_idx)], dtype=torch.long),
        "role_code": torch.tensor([int(row.role_code)], dtype=torch.int8),
        "is_radical": torch.tensor([int(row.radical)], dtype=torch.int8),
        "smiles_sanitized": torch.tensor([int(sanitized)], dtype=torch.int8),
    }
    legacy_x, legacy_edge = _legacy_features(molecule)
    ogb_x, ogb_edge = _ogb_features(molecule)
    legacy = add_random_walk_pe(
        Data(x=legacy_x, edge_attr=legacy_edge, **shared), walk_length=16
    )
    rich = Data(
        x=ogb_x,
        edge_index=edge_index,
        edge_attr=ogb_edge,
        y=shared["y"],
        source_idx=shared["source_idx"],
        role_code=shared["role_code"],
        is_radical=shared["is_radical"],
        smiles_sanitized=shared["smiles_sanitized"],
        random_walk_pe=legacy.random_walk_pe.clone(),
    )
    return {"legacy": legacy, "ogb": rich}


def _save_packed(path: Path, graphs: list[Data]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(graphs)
    atomic_torch(path, (data, slices))


def build_feature_screen_graph_shard(
    rows_dir: Path,
    graph_dir: Path,
    *,
    shard_index: int,
) -> dict:
    rows_dir, graph_dir = Path(rows_dir), Path(graph_dir)
    manifest = json.loads((rows_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("feature-screen rows are incomplete")
    source = manifest["shards"][shard_index]
    source_path = rows_dir / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("feature-screen source shard changed")
    report_path = graph_dir / "reports" / f"shard_{shard_index:04d}.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("builder_version") == SCREEN_BUILDER_VERSION
            and existing.get("source_sha256") == source["sha256"]
        ):
            for item in existing["files"]:
                path = graph_dir / item["path"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"feature-screen graph changed: {path}")
            return existing

    started = time.monotonic()
    frame = pd.read_csv(source_path)
    by_schema_role = {
        schema: {"train": [], "development": []} for schema in FEATURE_SCHEMAS
    }
    failures = []
    for row in frame.itertuples(index=False):
        try:
            graphs = _graphs_from_screen_row(row)
        except Exception as error:
            failures.append({
                "source_idx": int(row.source_idx),
                "error": type(error).__name__,
                "detail": str(error),
            })
            continue
        role = "train" if int(row.role_code) == 0 else "development"
        for schema, graph in graphs.items():
            by_schema_role[schema][role].append(graph)
    if failures:
        atomic_json(report_path, {
            "status": "failed",
            "builder_version": SCREEN_BUILDER_VERSION,
            "source_sha256": source["sha256"],
            "failures": failures,
        })
        raise RuntimeError(f"feature-screen graph shard has {len(failures)} failures")

    files = []
    for schema in FEATURE_SCHEMAS:
        for role in ("train", "development"):
            graphs = by_schema_role[schema][role]
            if not graphs:
                continue
            path = graph_dir / schema / role / f"{role}_shard_{shard_index:04d}.pt"
            _save_packed(path, graphs)
            files.append({
                "schema": schema,
                "role": role,
                "path": path.relative_to(graph_dir).as_posix(),
                "rows": len(graphs),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    report = {
        "format": "molgap-pcqm4mv2-feature-screen-graph-shard-v1",
        "status": "complete",
        "builder_version": SCREEN_BUILDER_VERSION,
        "shard_index": shard_index,
        "source_sha256": source["sha256"],
        "source_rows": source["rows"],
        "files": files,
        "failures": [],
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(report_path, report)
    return report


def accept_feature_screen_graphs(
    rows_dir: Path,
    graph_dir: Path,
    output_path: Path,
) -> dict:
    rows_dir, graph_dir, output_path = Path(rows_dir), Path(graph_dir), Path(output_path)
    manifest = json.loads((rows_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {}
    labels = {}
    radicals = {}
    for item in manifest["shards"]:
        frame = pd.read_csv(rows_dir / item["path"])
        for row in frame.itertuples(index=False):
            index = int(row.source_idx)
            expected[index] = int(row.role_code)
            labels[index] = float(row.gap)
            radicals[index] = int(row.radical)
    expected_by_role = {
        role: {index for index, code in expected.items() if code == role}
        for role in (0, 1)
    }
    reports = []
    schema_summary = {}
    for schema in FEATURE_SCHEMAS:
        seen = {0: set(), 1: set()}
        train_targets = []
        for item in manifest["shards"]:
            report_path = graph_dir / "reports" / f"shard_{item['shard_index']:04d}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "complete" or report.get("source_sha256") != item["sha256"]:
                raise RuntimeError(f"feature-screen report failed: {report_path}")
            for artifact in report["files"]:
                if artifact["schema"] != schema:
                    continue
                path = graph_dir / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"feature-screen graph hash failed: {path}")
                dataset = PackedGraphDataset(path)
                role_code = 0 if artifact["role"] == "train" else 1
                for graph in dataset:
                    index = int(graph.source_idx.item())
                    if index in seen[role_code] or expected.get(index) != role_code:
                        raise RuntimeError(f"feature-screen identity failed: {index}")
                    if not math.isclose(float(graph.y.item()), labels[index], abs_tol=1e-6):
                        raise RuntimeError(f"feature-screen target failed: {index}")
                    if int(graph.is_radical.item()) != radicals[index]:
                        raise RuntimeError(f"feature-screen radical flag failed: {index}")
                    if not torch.isfinite(graph.random_walk_pe).all():
                        raise RuntimeError(f"feature-screen RWSE failed: {index}")
                    if schema == "ogb":
                        if graph.x.dtype != torch.long or graph.edge_attr.dtype != torch.long:
                            raise RuntimeError("OGB features must be categorical long tensors")
                        for column, categories in enumerate(ATOM_FEATURE_DIMS):
                            if graph.x[:, column].min() < 0 or graph.x[:, column].max() >= categories:
                                raise RuntimeError("OGB atom category is out of range")
                        for column, categories in enumerate(BOND_FEATURE_DIMS):
                            if graph.edge_attr.numel() and (
                                graph.edge_attr[:, column].min() < 0
                                or graph.edge_attr[:, column].max() >= categories
                            ):
                                raise RuntimeError("OGB bond category is out of range")
                    seen[role_code].add(index)
                    if role_code == 0:
                        train_targets.append(float(graph.y.item()))
                del dataset
        if seen != expected_by_role:
            raise RuntimeError(f"feature-screen {schema} coverage differs")
        targets = np.asarray(train_targets, dtype=np.float64)
        schema_summary[schema] = {
            "counts": {"train": len(seen[0]), "development": len(seen[1])},
            "target_mean_gap": float(targets.mean()),
            "target_std_gap": float(targets.std()),
            "node_feature_fields": 27 if schema == "legacy" else len(ATOM_FEATURE_DIMS),
            "edge_feature_fields": 4 if schema == "legacy" else len(BOND_FEATURE_DIMS),
        }
    for item in manifest["shards"]:
        report_path = graph_dir / "reports" / f"shard_{item['shard_index']:04d}.json"
        reports.append({
            "path": report_path.relative_to(graph_dir).as_posix(),
            "sha256": sha256_file(report_path),
        })
    acceptance = {
        "format": "molgap-pcqm4mv2-feature-screen-acceptance-v1",
        "status": "accepted",
        "builder_version": SCREEN_BUILDER_VERSION,
        "rows_manifest_sha256": sha256_file(rows_dir / "manifest.json"),
        "official_valid_used": False,
        "official_test_used": False,
        "external_data_used": False,
        "atom_feature_dims": list(ATOM_FEATURE_DIMS),
        "bond_feature_dims": list(BOND_FEATURE_DIMS),
        "schemas": schema_summary,
        "reports": reports,
    }
    atomic_json(output_path, acceptance)
    return acceptance


def _graph_paths(graph_dir: Path, schema: str, role: str) -> list[Path]:
    paths = sorted((Path(graph_dir) / schema / role).glob(f"{role}_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {schema}/{role} feature-screen graphs")
    return paths


def _model(schema: str, config: FeatureScreenConfig) -> nn.Module:
    if config.model_family not in {"edge_state", "pair_gps"}:
        raise ValueError("model_family must be 'edge_state' or 'pair_gps'")
    if config.model_family == "pair_gps":
        if schema != "ogb":
            raise ValueError("pair_gps requires the OGB categorical schema")
        if config.graph_context != "none" or config.categorical_encoder != "sum":
            raise ValueError("pair_gps uses its fixed categorical input contract")
        return CategoricalPairGPS2DWrapper(
            atom_feature_dims=ATOM_FEATURE_DIMS,
            bond_feature_dims=BOND_FEATURE_DIMS,
            atom_input_channels=config.atom_input_channels,
            bond_input_channels=config.bond_input_channels,
            rwse_dim=config.rwse_dim,
            hidden_channels=config.hidden_channels,
            pair_channels=config.pair_channels,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            n_targets=1,
            path_steps=config.path_steps,
            triplet_rank=config.triplet_rank,
        )
    common = dict(
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        n_targets=1,
        rwse_dim=config.rwse_dim,
        edge_state_channels=config.edge_state_channels,
    )
    if schema == "legacy":
        return EdgeStateStructuralGPSWrapper(
            in_channels=len(OFFICIAL_ATOM_LIST) + 3,
            edge_dim=4,
            **common,
        )
    if schema == "ogb":
        if config.graph_context == "none":
            wrapper = CategoricalEdgeStateStructuralGPSWrapper
            context = {}
        elif config.graph_context == "radical":
            wrapper = CategoricalRadicalContextEdgeStateStructuralGPSWrapper
            context = {
                "radical_context_channels": config.radical_context_channels,
            }
        else:
            raise ValueError("graph_context must be 'none' or 'radical'")
        return wrapper(
            atom_feature_dims=ATOM_FEATURE_DIMS,
            bond_feature_dims=BOND_FEATURE_DIMS,
            categorical_encoder=config.categorical_encoder,
            categorical_field_channels=config.categorical_field_channels,
            **context,
            **common,
        )
    raise ValueError(f"unsupported feature schema: {schema}")


def _forward(model: nn.Module, batch) -> torch.Tensor:
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def preflight_feature_screen(
    graph_dir: Path,
    acceptance_path: Path,
    output_path: Path,
    *,
    batch_size: int = 32,
) -> dict:
    """Run one real accepted-graph CUDA forward/backward for both schemas."""
    if not torch.cuda.is_available():
        raise RuntimeError("feature-screen preflight requires CUDA")
    acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted":
        raise RuntimeError("feature-screen graph acceptance is missing")
    device = torch.device("cuda")
    config = FeatureScreenConfig()
    schemas = {}
    for schema in FEATURE_SCHEMAS:
        dataset = PackedGraphDataset(_graph_paths(graph_dir, schema, "train")[0])
        graphs = [dataset[index] for index in range(min(batch_size, len(dataset)))]
        batch = next(iter(DataLoader(graphs, batch_size=len(graphs)))).to(device)
        model = _model(schema, config).to(device)
        model.train()
        prediction = _forward(model, batch)
        loss = prediction.square().mean()
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not torch.isfinite(prediction).all() or not finite_gradients:
            raise RuntimeError(f"feature-screen {schema} CUDA preflight is non-finite")
        schemas[schema] = {
            "rows": len(graphs),
            "output_shape": list(prediction.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "finite_gradients": finite_gradients,
        }
        del batch, model, dataset
        torch.cuda.empty_cache()
    report = {
        "format": "molgap-pcqm-feature-screen-cuda-preflight-v1",
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "acceptance_sha256": sha256_file(Path(acceptance_path)),
        "schemas": schemas,
    }
    atomic_json(Path(output_path), report)
    return report


def preflight_feature_model(
    graph_dir: Path,
    acceptance_path: Path,
    output_path: Path,
    *,
    schema: str,
    config: FeatureScreenConfig,
    batch_size: int,
    training_rows: int = 100_000,
    measured_batches: int = 3,
) -> dict:
    """Measure one candidate on accepted graphs before starting a remote run."""
    if schema not in FEATURE_SCHEMAS:
        raise ValueError(f"unsupported feature schema: {schema}")
    if min(batch_size, training_rows, measured_batches) <= 0:
        raise ValueError("preflight row and batch counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("feature-model preflight requires CUDA")
    acceptance_path = Path(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted" or any(
        acceptance.get(key)
        for key in ("official_valid_used", "official_test_used", "external_data_used")
    ):
        raise RuntimeError("feature-screen acceptance is missing or contaminated")

    if config.precision not in {"amp", "fp32"}:
        raise ValueError("precision must be 'amp' or 'fp32'")
    device = torch.device("cuda")
    amp_enabled = config.precision == "amp"
    dataset = PackedGraphDataset(_graph_paths(graph_dir, schema, "train")[0])
    graphs = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs)))).to(device)
    model = _model(schema, config).to(device).train()

    def step() -> bool:
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction = _forward(model, batch)
            loss = prediction.square().mean()
        loss.backward()
        return bool(torch.isfinite(prediction).all()) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    finite = step()
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(measured_batches):
        finite = step() and finite
    torch.cuda.synchronize()
    seconds_per_batch = (time.monotonic() - started) / measured_batches
    projected_batches = math.ceil(training_rows / batch_size)
    projected_training_s = (
        seconds_per_batch * projected_batches * config.max_epochs
    )
    if not finite:
        raise RuntimeError("feature-model CUDA preflight is non-finite")
    report = {
        "format": "molgap-pcqm-feature-model-preflight-v1",
        "status": "passed",
        "schema": schema,
        "config": asdict(config),
        "rows": len(graphs),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds_per_batch": seconds_per_batch,
        "projected_training_s": projected_training_s,
        "projected_training_hours": projected_training_s / 3600.0,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "finite_gradients": True,
        "acceptance_sha256": sha256_file(acceptance_path),
    }
    atomic_json(Path(output_path), report)
    return report


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    graph_dir: Path,
    schema: str,
    device: torch.device,
    batch_size: int,
    mean: float,
    std: float,
    amp_enabled: bool,
) -> dict:
    model.eval()
    source_idx, prediction, target, radical = [], [], [], []
    for path in _graph_paths(graph_dir, schema, "development"):
        dataset = PackedGraphDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda" and amp_enabled
            ):
                values = _forward(model, batch).float() * std + mean
            if not torch.isfinite(values).all():
                raise FloatingPointError("feature-screen evaluation became non-finite")
            source_idx.append(batch.source_idx.view(-1).cpu())
            prediction.append(values.cpu())
            target.append(batch.y.view(-1).cpu())
            radical.append(batch.is_radical.view(-1).bool().cpu())
        del loader, dataset
    payload = {
        "source_idx": torch.cat(source_idx),
        "prediction_eV": torch.cat(prediction),
        "target_eV": torch.cat(target),
        "is_radical": torch.cat(radical),
    }
    order = torch.argsort(payload["source_idx"])
    payload = {key: value[order] for key, value in payload.items()}
    absolute = (payload["prediction_eV"] - payload["target_eV"]).abs()
    mask = payload["is_radical"]
    return {
        "payload": payload,
        "mae_eV": float(absolute.mean()),
        "radical_mae_eV": float(absolute[mask].mean()),
        "nonradical_mae_eV": float(absolute[~mask].mean()),
        "radical_rows": int(mask.sum()),
        "nonradical_rows": int((~mask).sum()),
    }


def train_feature_screen(
    graph_dir: Path,
    acceptance_path: Path,
    output_dir: Path,
    *,
    schema: str,
    seed: int,
    config: FeatureScreenConfig = FeatureScreenConfig(),
) -> dict:
    if schema not in FEATURE_SCHEMAS:
        raise ValueError(f"unsupported feature schema: {schema}")
    if not torch.cuda.is_available():
        raise RuntimeError("feature-screen training requires CUDA")
    if config.precision not in {"amp", "fp32"}:
        raise ValueError("precision must be 'amp' or 'fp32'")
    started = time.monotonic()
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_enabled = config.precision == "amp"
    acceptance_path = Path(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted" or any(
        acceptance.get(key) for key in ("official_valid_used", "official_test_used", "external_data_used")
    ):
        raise RuntimeError("feature-screen acceptance is missing or contaminated")
    summary = acceptance["schemas"][schema]
    mean = float(summary["target_mean_gap"])
    std = float(summary["target_std_gap"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _model(schema, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.L1Loss()
    contract = {**asdict(config), "schema": schema, "seed": seed}
    acceptance_hash = sha256_file(acceptance_path)
    best_mae, best_epoch, wait, start_epoch = float("inf"), -1, 0, 0
    log = []
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("contract") != contract or state.get("acceptance_sha256") != acceptance_hash:
            raise RuntimeError("feature-screen resume contract changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_mae = float(state["best_mae_eV"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])
        start_epoch = int(state["epoch"]) + 1
        log = list(state["log"])

    base_paths = _graph_paths(graph_dir, schema, "train")
    for epoch in range(start_epoch, config.max_epochs):
        if time.monotonic() - started >= config.hard_job_budget_s:
            break
        epoch_started = time.monotonic()
        model.train()
        paths = list(base_paths)
        random.Random(seed + epoch).shuffle(paths)
        loss_sum, rows = 0.0, 0
        for shard_number, path in enumerate(paths):
            dataset = PackedGraphDataset(path)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed * 100_000 + epoch * 100 + shard_number),
                pin_memory=True,
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    prediction = _forward(model, batch)
                    target = (batch.y.view(-1) - mean) / std
                    loss = criterion(prediction, target)
                if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"feature-screen training became non-finite at epoch {epoch}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"feature-screen gradients became non-finite at epoch {epoch}"
                    )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * batch.num_graphs
                rows += int(batch.num_graphs)
            del loader, dataset
            gc.collect()
        scheduler.step()
        evaluation = _evaluate(
            model,
            graph_dir,
            schema,
            device,
            config.eval_batch_size,
            mean,
            std,
            amp_enabled,
        )
        improved = np.isfinite(evaluation["mae_eV"]) and evaluation["mae_eV"] < best_mae
        if improved:
            best_mae, best_epoch, wait = evaluation["mae_eV"], epoch, 0
            atomic_torch(best_path, {
                "format": "molgap-pcqm-feature-screen-best-v1",
                "contract": contract,
                "model": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch,
                "best_mae_eV": best_mae,
                "target_mean_gap": mean,
                "target_std_gap": std,
                "acceptance_sha256": acceptance_hash,
            })
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_l1_normalized": loss_sum / max(rows, 1),
            "train_rows": rows,
            "development_mae_eV": evaluation["mae_eV"],
            "radical_mae_eV": evaluation["radical_mae_eV"],
            "nonradical_mae_eV": evaluation["nonradical_mae_eV"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": time.monotonic() - epoch_started,
            "selected": bool(improved),
        }
        log.append(row)
        atomic_torch(last_path, {
            "format": "molgap-pcqm-feature-screen-checkpoint-v1",
            "contract": contract,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae_eV": best_mae,
            "best_epoch": best_epoch,
            "wait": wait,
            "log": log,
            "acceptance_sha256": acceptance_hash,
        })
        atomic_json(output_dir / "progress.json", {
            "status": "training",
            "schema": schema,
            "seed": seed,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_mae_eV": best_mae,
        })
        print(
            f"{schema} seed={seed} ep{epoch:02d} train={row['train_l1_normalized']:.6f} "
            f"dev={row['development_mae_eV']:.6f} radical={row['radical_mae_eV']:.6f} "
            f"nonradical={row['nonradical_mae_eV']:.6f} {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break

    if not best_path.is_file():
        raise RuntimeError("feature-screen training selected no model")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    evaluation = _evaluate(
        model,
        graph_dir,
        schema,
        device,
        config.eval_batch_size,
        mean,
        std,
        amp_enabled,
    )
    atomic_torch(output_dir / "development_predictions.pt", evaluation.pop("payload"))
    metrics = {
        "format": "molgap-pcqm-feature-screen-training-v1",
        "status": "complete",
        "schema": schema,
        "seed": seed,
        "config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best["best_epoch"]),
        "development": evaluation,
        "train_log": log,
        "official_train_only": True,
        "official_valid_used": False,
        "official_test_used": False,
        "external_data_used": False,
        "runtime_s": time.monotonic() - started,
        "best_sha256": sha256_file(best_path),
        "predictions_sha256": sha256_file(output_dir / "development_predictions.pt"),
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(output_dir / "completion_manifest.json", {
        "status": "complete",
        "best": {"path": "best.pt", "sha256": metrics["best_sha256"]},
        "metrics": {"path": "metrics.json", "sha256": sha256_file(output_dir / "metrics.json")},
        "predictions": {"path": "development_predictions.pt", "sha256": metrics["predictions_sha256"]},
    })
    return metrics


def accept_feature_screen_runs(runs_dir: Path, output_path: Path) -> dict:
    runs_dir, output_path = Path(runs_dir), Path(output_path)
    runs = {}
    references = {}
    comparisons = []
    for seed in (42, 43, 44):
        runs[str(seed)] = {}
        for schema in FEATURE_SCHEMAS:
            root = runs_dir / f"{schema}_seed{seed}"
            metrics_path = root / "metrics.json"
            predictions_path = root / "development_predictions.pt"
            manifest = json.loads((root / "completion_manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or metrics.get("status") != "complete":
                raise RuntimeError(f"incomplete feature-screen run: {root}")
            if sha256_file(metrics_path) != manifest["metrics"]["sha256"]:
                raise RuntimeError(f"feature-screen metrics hash failed: {root}")
            if sha256_file(predictions_path) != manifest["predictions"]["sha256"]:
                raise RuntimeError(f"feature-screen predictions hash failed: {root}")
            payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
            if seed not in references:
                references[seed] = payload
            else:
                for key in ("source_idx", "target_eV", "is_radical"):
                    if not torch.equal(payload[key], references[seed][key]):
                        raise RuntimeError(f"feature-screen paired identity failed: seed {seed}")
            runs[str(seed)][schema] = metrics
        legacy = runs[str(seed)]["legacy"]["development"]
        rich = runs[str(seed)]["ogb"]["development"]
        comparisons.append({
            "seed": seed,
            "overall_delta_eV": rich["mae_eV"] - legacy["mae_eV"],
            "radical_delta_eV": rich["radical_mae_eV"] - legacy["radical_mae_eV"],
            "nonradical_delta_eV": rich["nonradical_mae_eV"] - legacy["nonradical_mae_eV"],
        })
    gates = {
        "overall_improvement_each_seed_eV": 0.010,
        "radical_improvement_each_seed_eV": 0.10,
        "maximum_nonradical_regression_each_seed_eV": 0.002,
    }
    passed = all(
        item["overall_delta_eV"] <= -gates["overall_improvement_each_seed_eV"]
        and item["radical_delta_eV"] <= -gates["radical_improvement_each_seed_eV"]
        and item["nonradical_delta_eV"] <= gates["maximum_nonradical_regression_each_seed_eV"]
        for item in comparisons
    )
    report = {
        "format": "molgap-pcqm-feature-screen-multiseed-acceptance-v1",
        "status": "accepted",
        "runs": runs,
        "comparisons": comparisons,
        "aggregate": {
            key: float(np.mean([item[key] for item in comparisons]))
            for key in ("overall_delta_eV", "radical_delta_eV", "nonradical_delta_eV")
        },
        "gates": gates,
        "passed": passed,
        "official_valid_used": False,
        "official_test_used": False,
    }
    atomic_json(output_path, report)
    return report


def accept_schedule_screen_runs(runs_dir: Path, output_path: Path) -> dict:
    """Accept a paired three-seed collapse-vs-warmup schedule screen."""
    runs_dir, output_path = Path(runs_dir), Path(output_path)
    runs: dict[str, dict[str, dict]] = {}
    comparisons = []
    references = {}
    for seed in (42, 43, 44):
        runs[str(seed)] = {}
        for variant in SCHEDULE_VARIANTS:
            root = runs_dir / f"{variant}_seed{seed}"
            metrics_path = root / "metrics.json"
            predictions_path = root / "development_predictions.pt"
            manifest = json.loads(
                (root / "completion_manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or metrics.get("status") != "complete":
                raise RuntimeError(f"incomplete schedule-screen run: {root}")
            if metrics.get("schema") != "ogb" or any(
                metrics.get(key)
                for key in ("official_valid_used", "official_test_used", "external_data_used")
            ):
                raise RuntimeError(f"invalid schedule-screen contract: {root}")
            if sha256_file(metrics_path) != manifest["metrics"]["sha256"]:
                raise RuntimeError(f"schedule-screen metrics hash failed: {root}")
            if sha256_file(predictions_path) != manifest["predictions"]["sha256"]:
                raise RuntimeError(f"schedule-screen predictions hash failed: {root}")
            payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
            if seed not in references:
                references[seed] = payload
            else:
                for key in ("source_idx", "target_eV", "is_radical"):
                    if not torch.equal(payload[key], references[seed][key]):
                        raise RuntimeError(f"schedule-screen paired identity failed: seed {seed}")
            runs[str(seed)][variant] = metrics
        baseline = runs[str(seed)]["collapse10"]["development"]
        candidate = runs[str(seed)]["warmup20"]["development"]
        comparisons.append({
            "seed": seed,
            "overall_delta_eV": candidate["mae_eV"] - baseline["mae_eV"],
            "radical_delta_eV": candidate["radical_mae_eV"] - baseline["radical_mae_eV"],
            "nonradical_delta_eV": (
                candidate["nonradical_mae_eV"] - baseline["nonradical_mae_eV"]
            ),
        })
    mean_delta = float(np.mean([item["overall_delta_eV"] for item in comparisons]))
    improved_seeds = sum(item["overall_delta_eV"] < 0.0 for item in comparisons)
    maximum_regression = max(item["overall_delta_eV"] for item in comparisons)
    gates = {
        "minimum_mean_improvement_eV": 0.001,
        "minimum_improved_seeds": 2,
        "maximum_single_seed_regression_eV": 0.002,
    }
    passed = (
        mean_delta <= -gates["minimum_mean_improvement_eV"]
        and improved_seeds >= gates["minimum_improved_seeds"]
        and maximum_regression <= gates["maximum_single_seed_regression_eV"]
    )
    report = {
        "format": "molgap-pcqm-schedule-screen-multiseed-acceptance-v1",
        "status": "accepted",
        "runs": runs,
        "comparisons": comparisons,
        "aggregate": {
            "overall_delta_eV": mean_delta,
            "radical_delta_eV": float(
                np.mean([item["radical_delta_eV"] for item in comparisons])
            ),
            "nonradical_delta_eV": float(
                np.mean([item["nonradical_delta_eV"] for item in comparisons])
            ),
            "improved_seeds": improved_seeds,
            "maximum_single_seed_regression_eV": maximum_regression,
        },
        "gates": gates,
        "passed": passed,
        "official_valid_used": False,
        "official_test_used": False,
    }
    atomic_json(output_path, report)
    return report
