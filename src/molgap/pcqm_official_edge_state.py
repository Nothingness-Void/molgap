"""Official-only PCQM4Mv2 EdgeState training and OGB submission artifacts.

The module deliberately keeps test SMILES out of graph construction until a
model has been selected on the official validation split. CPU graph shards,
GPU checkpoints, and final test predictions are independently resumable.
"""
from __future__ import annotations

import copy
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ogb
from ogb.lsc import PCQM4Mv2Evaluator
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .gps import (
    CategoricalEdgeStateStructuralGPSWrapper,
    EdgeStateStructuralGPSWrapper,
)
from .ogb_features import (
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    atom_to_ogb_feature_vector,
    bond_to_ogb_feature_vector,
)
from .structural_encoding import add_random_walk_pe


CSV_MEMBER = "pcqm4m-v2/raw/data.csv.gz"
SPLIT_MEMBER = "pcqm4m-v2/split_dict.pt"
SPLIT_NAMES = ("train", "valid", "test-dev", "test-challenge")
TRAINING_SPLITS = {"train": 0, "valid": 1}
OFFICIAL_ATOM_LIST = (
    6, 7, 8, 9, 16, 17, 15, 35, 14, 5, 34, 32, 33, 12, 2,
    1, 31, 30, 13, 22, 20, 36, 18, 4,
)
FEATURE_SCHEMAS = ("legacy", "ogb")
GRAPH_BUILDER_VERSIONS = {
    "legacy": "official-complete-elements-topology-fallback-v3",
    "ogb": "official-ogb-categorical-topology-fallback-v1",
}


@dataclass(frozen=True)
class OfficialEdgeStateConfig:
    hidden_channels: int = 192
    num_layers: int = 9
    num_heads: int = 4
    dropout: float = 0.05
    rwse_dim: int = 16
    edge_state_channels: int = 64
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 4.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 10
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    minimum_learning_rate: float = 1.0e-6
    feature_schema: str = "legacy"
    projection_epochs: int = 0
    patience: int = 3
    seed: int = 42
    gradient_clip: float = 1.0
    max_projected_training_s: float = 12.0 * 3600.0
    hard_job_budget_s: float = 11.5 * 3600.0


@dataclass(frozen=True)
class OfficialEdgeStateContinuationConfig:
    """A separately recorded schedule for continuing a frozen full run."""

    additional_epochs: int = 20
    learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 2
    patience: int = 7
    batch_size: int = 256
    eval_batch_size: int = 512
    gradient_clip: float = 1.0
    hard_job_budget_s: float = 13.5 * 3600.0


class PackedGraphDataset(InMemoryDataset):
    def __init__(self, path: Path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(
            path, map_location="cpu", weights_only=False
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, y_pred: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, y_pred=np.asarray(y_pred, dtype=np.float32))
    os.replace(temporary, path)


def atomic_ogb_submission(
    output_dir: Path,
    mode: str,
    y_pred: np.ndarray,
) -> Path:
    """Write one submission through OGB's evaluator, then move atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    evaluator = PCQM4Mv2Evaluator()
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        evaluator.save_test_submission(
            {"y_pred": values},
            temporary,
            mode=mode,
        )
        generated = Path(temporary) / f"y_pred_pcqm4m-v2_{mode}.npz"
        if not generated.is_file():
            raise RuntimeError(f"OGB evaluator did not create {generated.name}")
        destination = output_dir / generated.name
        os.replace(generated, destination)
    return destination


def validate_submission_files(output_dir: Path) -> dict:
    """Validate OGB filenames, keys, shapes, dtypes, finiteness, and hashes."""
    output_dir = Path(output_dir)
    expected_rows = {"test-dev": 147_037, "test-challenge": 147_432}
    outputs = {}
    for mode, rows in expected_rows.items():
        path = output_dir / f"y_pred_pcqm4m-v2_{mode}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ["y_pred"]:
                raise RuntimeError(f"Unexpected NPZ keys in {path}: {payload.files}")
            prediction = payload["y_pred"]
        if prediction.shape != (rows,):
            raise RuntimeError(
                f"Unexpected {mode} prediction shape: {prediction.shape} != {(rows,)}"
            )
        if prediction.dtype != np.float32:
            raise RuntimeError(f"Unexpected {mode} dtype: {prediction.dtype}")
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite values in {mode} prediction")
        outputs[mode] = {
            "path": path.name,
            "rows": rows,
            "dtype": "float32",
            "sha256": sha256_file(path),
        }
    result = {
        "format": "pcqm4mv2-ogb-submission-acceptance-v1",
        "status": "accepted",
        "ogb_version": ogb.__version__,
        "minimum_ogb_version": "1.3.2",
        "outputs": outputs,
    }
    manifest_path = output_dir / "submission_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError("Submission manifest is not complete")
        if manifest.get("resumed") is not False:
            raise RuntimeError("Official timing requires one clean, non-resumed run")
        if manifest.get("official_4h_timing_pass") is not True:
            raise RuntimeError("Official four-hour inference timing gate did not pass")
        manifest_outputs = manifest.get("outputs", {})
        for mode, accepted in outputs.items():
            recorded = manifest_outputs.get(mode, {})
            if recorded.get("sha256") != accepted["sha256"]:
                raise RuntimeError(f"Submission manifest hash changed for {mode}")
        result["timing"] = {
            "elapsed_s": float(manifest["elapsed_s"]),
            "limit_s": float(manifest["inference_limit_s"]),
            "official_4h_timing_pass": True,
            "single_gpu": bool(manifest.get("single_gpu")),
            "cpu_workers": int(manifest.get("cpu_workers", 0)),
        }
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_official_splits(archive: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(archive) as bundle:
        if SPLIT_MEMBER not in bundle.namelist():
            raise FileNotFoundError(f"{SPLIT_MEMBER} is absent from {archive}")
        raw = torch.load(
            io.BytesIO(bundle.read(SPLIT_MEMBER)),
            map_location="cpu",
            weights_only=False,
        )
    if set(raw) != set(SPLIT_NAMES):
        raise ValueError(f"Unexpected official split keys: {sorted(raw)}")
    return {
        name: np.asarray(raw[name], dtype=np.int64).reshape(-1)
        for name in SPLIT_NAMES
    }


def validate_official_splits(splits: dict[str, np.ndarray]) -> dict:
    arrays = [splits[name] for name in SPLIT_NAMES]
    total = sum(len(values) for values in arrays)
    combined = np.concatenate(arrays)
    if total == 0 or len(np.unique(combined)) != total:
        raise ValueError("Official PCQM4Mv2 splits overlap or are empty")
    if int(combined.min()) != 0 or int(combined.max()) != total - 1:
        raise ValueError("Official PCQM4Mv2 split indices are not a full row partition")
    return {
        "rows": total,
        "counts": {name: int(len(splits[name])) for name in SPLIT_NAMES},
        "minimum_idx": int(combined.min()),
        "maximum_idx": int(combined.max()),
    }


def _split_lookup(splits: dict[str, np.ndarray]) -> np.ndarray:
    contract = validate_official_splits(splits)
    lookup = np.full(contract["rows"], -1, dtype=np.int8)
    for code, name in enumerate(SPLIT_NAMES):
        lookup[splits[name]] = code
    if bool((lookup < 0).any()):
        raise RuntimeError("Official split lookup has uncovered source rows")
    return lookup


def _open_official_csv(archive: Path):
    bundle = zipfile.ZipFile(archive)
    if CSV_MEMBER not in bundle.namelist():
        bundle.close()
        raise FileNotFoundError(f"{CSV_MEMBER} is absent from {archive}")
    compressed = bundle.open(CSV_MEMBER)
    stream = gzip.GzipFile(fileobj=compressed)
    return bundle, compressed, stream


def prepare_training_rows(
    archive: Path,
    output_dir: Path,
    *,
    source_shard_rows: int = 50_000,
) -> dict:
    """Materialize only official train/valid rows into immutable CSV shards."""
    archive = Path(archive)
    output_dir = Path(output_dir)
    if source_shard_rows <= 0:
        raise ValueError("source_shard_rows must be positive")
    archive_hash = sha256_file(archive)
    splits = load_official_splits(archive)
    split_contract = validate_official_splits(splits)
    lookup = _split_lookup(splits)
    manifest_path = output_dir / "manifest.json"
    existing = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("archive_sha256") == archive_hash
            and existing.get("source_shard_rows") == source_shard_rows
        ):
            for item in existing["shards"]:
                path = output_dir / item["path"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"Prepared source shard changed: {path}")
            return existing
        if (
            existing.get("archive_sha256") != archive_hash
            or existing.get("source_shard_rows") != source_shard_rows
        ):
            raise RuntimeError("Official row preparation resume contract changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = existing or {
        "format": "molgap-pcqm4mv2-official-training-rows-v1",
        "status": "preparing",
        "archive_path": str(archive),
        "archive_sha256": archive_hash,
        "csv_member": CSV_MEMBER,
        "split_member": SPLIT_MEMBER,
        "split_contract": split_contract,
        "source_shard_rows": int(source_shard_rows),
        "rows_seen": 0,
        "training_rows": 0,
        "counts": {"train": 0, "valid": 0},
        "shards": [],
        "test_smiles_materialized": False,
        "external_data_used": False,
    }
    atomic_json(manifest_path, manifest)

    bundle, compressed, stream = _open_official_csv(archive)
    try:
        reader = pd.read_csv(stream, chunksize=source_shard_rows)
        for chunk_index, frame in enumerate(reader):
            if list(frame.columns) != ["idx", "smiles", "homolumogap"]:
                raise ValueError(f"Unexpected official CSV columns: {list(frame.columns)}")
            start = chunk_index * source_shard_rows
            end = start + len(frame)
            expected = np.arange(start, end, dtype=np.int64)
            actual = frame["idx"].to_numpy(dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise ValueError(f"Official CSV idx sequence changed at row {start}")
            codes = lookup[start:end]
            training_mask = codes <= 1
            selected = frame.loc[training_mask, ["idx", "smiles", "homolumogap"]].copy()
            selected.rename(columns={"homolumogap": "gap"}, inplace=True)
            selected["split_code"] = codes[training_mask]
            if not np.isfinite(selected["gap"].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"Official train/valid label is non-finite in rows {start}:{end}")
            counts = {
                "train": int((selected["split_code"] == 0).sum()),
                "valid": int((selected["split_code"] == 1).sum()),
            }
            shard_path = output_dir / f"source_{start:07d}_{end:07d}.csv.gz"
            prior = next(
                (item for item in manifest["shards"] if item["path"] == shard_path.name),
                None,
            )
            if prior is None:
                temporary = shard_path.with_name(f".{shard_path.name}.tmp.gz")
                selected.to_csv(temporary, index=False, compression="gzip")
                os.replace(temporary, shard_path)
                report = {
                    "path": shard_path.name,
                    "source_start": start,
                    "source_end": end,
                    "rows": int(len(selected)),
                    "counts": counts,
                    "bytes": shard_path.stat().st_size,
                    "sha256": sha256_file(shard_path),
                }
                manifest["shards"].append(report)
                manifest["rows_seen"] = end
                manifest["training_rows"] += int(len(selected))
                for role, count in counts.items():
                    manifest["counts"][role] += count
                atomic_json(manifest_path, manifest)
            elif not shard_path.is_file() or sha256_file(shard_path) != prior["sha256"]:
                raise RuntimeError(f"Prepared source shard changed: {shard_path}")
            print(
                f"official rows {start:,}:{end:,} train={counts['train']:,} "
                f"valid={counts['valid']:,}",
                flush=True,
            )
    finally:
        stream.close()
        compressed.close()
        bundle.close()

    if manifest["rows_seen"] != split_contract["rows"]:
        raise RuntimeError("Official CSV and split row counts differ")
    if manifest["counts"] != {
        "train": split_contract["counts"]["train"],
        "valid": split_contract["counts"]["valid"],
    }:
        raise RuntimeError("Prepared train/valid split counts changed")
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    return manifest


_WORKER_ATOMS = OFFICIAL_ATOM_LIST
_WORKER_RWSE = 16
_WORKER_FEATURE_SCHEMA = "legacy"


def _init_graph_worker(
    atom_list: tuple[int, ...],
    rwse_dim: int,
    feature_schema: str = "legacy",
) -> None:
    global _WORKER_ATOMS, _WORKER_RWSE, _WORKER_FEATURE_SCHEMA
    if feature_schema not in FEATURE_SCHEMAS:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    _WORKER_ATOMS = tuple(atom_list)
    _WORKER_RWSE = int(rwse_dim)
    _WORKER_FEATURE_SCHEMA = feature_schema
    torch.set_num_threads(1)


def _graph_from_row(row: tuple[int, str, float, int]) -> tuple[Data | None, dict | None]:
    source_idx, smiles, gap, split_code = row
    molecule = Chem.MolFromSmiles(smiles)
    smiles_sanitized = True
    if molecule is None:
        # A small official subset contains hypervalent silicon strings that
        # older RDKit releases accepted. Their bond topology remains usable;
        # retaining it is stricter than silently dropping official rows.
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        smiles_sanitized = False
    if molecule is None:
        return None, {"source_idx": source_idx, "reason": "unparseable_smiles"}
    if not smiles_sanitized:
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
    if molecule.GetNumAtoms() == 0:
        return None, {"source_idx": source_idx, "reason": "no_atoms"}
    rows, columns, edge_features = [], [], []
    if _WORKER_FEATURE_SCHEMA == "ogb":
        node_features = [
            atom_to_ogb_feature_vector(atom) for atom in molecule.GetAtoms()
        ]
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = bond_to_ogb_feature_vector(bond)
            rows.extend((left, right))
            columns.extend((right, left))
            edge_features.extend((feature, feature))
        node_tensor = torch.tensor(node_features, dtype=torch.long)
        edge_tensor = (
            torch.tensor(edge_features, dtype=torch.long)
            if edge_features
            else torch.zeros((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)
        )
    else:
        atom_to_index = {value: index for index, value in enumerate(_WORKER_ATOMS)}
        unsupported = sorted({
            atom.GetAtomicNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() not in atom_to_index
        })
        if unsupported:
            return None, {
                "source_idx": source_idx,
                "reason": "unsupported_atomic_number",
                "atomic_numbers": unsupported,
            }
        node_features = []
        for atom in molecule.GetAtoms():
            one_hot = [0.0] * len(_WORKER_ATOMS)
            one_hot[atom_to_index[atom.GetAtomicNum()]] = 1.0
            node_features.append(
                one_hot
                + [
                    atom.GetDegree() / 4.0,
                    atom.GetFormalCharge() / 2.0,
                    float(atom.GetIsAromatic()),
                ]
            )
        bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }
        for bond in molecule.GetBonds():
            if bond.GetBondType() not in bond_types:
                return None, {
                    "source_idx": source_idx,
                    "reason": "unsupported_bond_type",
                }
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond_types[bond.GetBondType()]
            feature = [float(bond_type == index) for index in range(4)]
            rows.extend((left, right))
            columns.extend((right, left))
            edge_features.extend((feature, feature))
        node_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_tensor = (
            torch.tensor(edge_features, dtype=torch.float32)
            if edge_features
            else torch.zeros((0, 4), dtype=torch.float32)
        )
    graph = Data(
        x=node_tensor,
        edge_index=torch.tensor([rows, columns], dtype=torch.long),
        edge_attr=edge_tensor,
        y=torch.tensor([gap], dtype=torch.float32),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
        split_code=torch.tensor([split_code], dtype=torch.int8),
        smiles_sanitized=torch.tensor([int(smiles_sanitized)], dtype=torch.int8),
    )
    try:
        return add_random_walk_pe(graph, walk_length=_WORKER_RWSE), None
    except Exception as error:
        return None, {
            "source_idx": source_idx,
            "reason": "rwse_failure",
            "detail": type(error).__name__,
        }


def _save_packed(path: Path, graphs: list[Data]) -> None:
    if not graphs:
        return
    data, slices = InMemoryDataset.collate(graphs)
    atomic_torch(path, (data, slices))


def _prepared_manifest(rows_dir: Path) -> dict:
    path = Path(rows_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Official train/valid row preparation is incomplete")
    return manifest


def build_training_graph_shard(
    rows_dir: Path,
    graph_dir: Path,
    *,
    shard_index: int,
    workers: int = 1,
    atom_list: tuple[int, ...] = OFFICIAL_ATOM_LIST,
    rwse_dim: int = 16,
    feature_schema: str = "legacy",
) -> dict:
    """Build one independently accepted train/valid graph shard."""
    if feature_schema not in FEATURE_SCHEMAS:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    builder_version = GRAPH_BUILDER_VERSIONS[feature_schema]
    rows_dir, graph_dir = Path(rows_dir), Path(graph_dir)
    manifest = _prepared_manifest(rows_dir)
    if shard_index < 0 or shard_index >= len(manifest["shards"]):
        raise IndexError(f"shard_index {shard_index} is outside prepared rows")
    source = manifest["shards"][shard_index]
    source_path = rows_dir / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise RuntimeError(f"Prepared source shard changed: {source_path}")
    report_path = graph_dir / "reports" / f"shard_{shard_index:04d}.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("source_sha256") == source["sha256"]
            and existing.get("feature_schema", "legacy") == feature_schema
            and (
                feature_schema == "ogb"
                or existing.get("atom_list") == list(atom_list)
            )
            and existing.get("rwse_dim") == rwse_dim
            and existing.get("graph_builder_version") == builder_version
            and not existing.get("failures")
        ):
            for item in existing["files"]:
                path = graph_dir / item["path"]
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise RuntimeError(f"Graph shard changed: {path}")
            return existing

    frame = pd.read_csv(source_path)
    required = ["idx", "smiles", "gap", "split_code"]
    if list(frame.columns) != required:
        raise ValueError(f"Unexpected prepared row columns: {list(frame.columns)}")
    payload = [
        (int(row.idx), str(row.smiles), float(row.gap), int(row.split_code))
        for row in frame.itertuples(index=False)
    ]
    started = time.monotonic()
    if workers > 1:
        context = mp.get_context("spawn" if os.name == "nt" else "fork")
        with context.Pool(
            processes=workers,
            initializer=_init_graph_worker,
            initargs=(tuple(atom_list), rwse_dim, feature_schema),
        ) as pool:
            results = list(pool.imap(_graph_from_row, payload, chunksize=128))
    else:
        _init_graph_worker(tuple(atom_list), rwse_dim, feature_schema)
        results = [_graph_from_row(row) for row in payload]

    by_role = {"train": [], "valid": []}
    failures = []
    unsanitized_fallback = []
    for (graph, failure), row in zip(results, payload):
        if failure is not None:
            failures.append(failure)
            continue
        if not bool(graph.smiles_sanitized.item()):
            unsanitized_fallback.append(int(row[0]))
        role = "train" if row[3] == 0 else "valid"
        by_role[role].append(graph)
    graph_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for role, graphs in by_role.items():
        if not graphs:
            continue
        path = graph_dir / role / f"{role}_shard_{shard_index:04d}.pt"
        _save_packed(path, graphs)
        files.append({
            "role": role,
            "path": path.relative_to(graph_dir).as_posix(),
            "rows": len(graphs),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    report = {
        "format": "molgap-pcqm4mv2-official-edge-state-graph-shard-v1",
        "status": "complete",
        "graph_builder_version": builder_version,
        "feature_schema": feature_schema,
        "shard_index": int(shard_index),
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "source_rows": int(source["rows"]),
        "atom_list": list(atom_list),
        "node_feature_dim": (
            len(ATOM_FEATURE_DIMS) if feature_schema == "ogb" else len(atom_list) + 3
        ),
        "edge_feature_dim": (
            len(BOND_FEATURE_DIMS) if feature_schema == "ogb" else 4
        ),
        "atom_feature_dims": (
            list(ATOM_FEATURE_DIMS) if feature_schema == "ogb" else None
        ),
        "bond_feature_dims": (
            list(BOND_FEATURE_DIMS) if feature_schema == "ogb" else None
        ),
        "rwse_dim": int(rwse_dim),
        "files": files,
        "counts": {role: len(graphs) for role, graphs in by_role.items()},
        "failures": failures,
        "unsanitized_fallback_source_idx": unsanitized_fallback,
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(report_path, report)
    return report


def _packed_data(path: Path):
    dataset = PackedGraphDataset(path)
    return dataset, dataset._data


def accept_training_graphs(
    archive: Path,
    rows_dir: Path,
    graph_dir: Path,
    output_path: Path,
    *,
    feature_schema: str = "legacy",
) -> dict:
    """Verify complete identity, labels, features, and hashes before GPU use."""
    if feature_schema not in FEATURE_SCHEMAS:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    builder_version = GRAPH_BUILDER_VERSIONS[feature_schema]
    archive, rows_dir, graph_dir = Path(archive), Path(rows_dir), Path(graph_dir)
    source_manifest = _prepared_manifest(rows_dir)
    if source_manifest["archive_sha256"] != sha256_file(archive):
        raise RuntimeError("Official archive changed after row preparation")
    splits = load_official_splits(archive)
    lookup = _split_lookup(splits)
    seen = np.zeros(len(lookup), dtype=np.bool_)
    counts = {"train": 0, "valid": 0}
    label_sum = 0.0
    label_squared_sum = 0.0
    reports = []
    failures = []
    unsanitized_fallback = []
    for shard_index, source in enumerate(source_manifest["shards"]):
        report_path = graph_dir / "reports" / f"shard_{shard_index:04d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or report["source_sha256"] != source["sha256"]
            or report.get("feature_schema", "legacy") != feature_schema
            or report.get("graph_builder_version") != builder_version
        ):
            raise RuntimeError(f"Graph report contract failed: {report_path}")
        failures.extend(report["failures"])
        unsanitized_fallback.extend(report.get("unsanitized_fallback_source_idx", []))
        expected_rows = pd.read_csv(rows_dir / source["path"]).set_index("idx")
        for item in report["files"]:
            path = graph_dir / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Graph shard hash failed: {path}")
            dataset, data = _packed_data(path)
            if len(dataset) != item["rows"]:
                raise RuntimeError(f"Graph shard row count failed: {path}")
            indices = data.source_idx.view(-1).numpy().astype(np.int64)
            if bool(seen[indices].any()):
                raise RuntimeError(f"Duplicate graph source_idx in {path}")
            seen[indices] = True
            expected_code = 0 if item["role"] == "train" else 1
            if not bool((lookup[indices] == expected_code).all()):
                raise RuntimeError(f"Official split identity failed: {path}")
            if not bool((data.split_code.view(-1).numpy() == expected_code).all()):
                raise RuntimeError(f"Stored split code failed: {path}")
            if data.smiles_sanitized.numel() != len(dataset):
                raise RuntimeError(f"SMILES parse-mode ledger failed: {path}")
            targets = data.y.view(-1).numpy().astype(np.float64)
            expected_targets = expected_rows.loc[indices, "gap"].to_numpy(dtype=np.float64)
            if not np.allclose(targets, expected_targets, atol=1.0e-6, rtol=0.0):
                raise RuntimeError(f"Official labels changed in {path}")
            if (
                data.x.shape[1] != report["node_feature_dim"]
                or data.edge_attr.shape[1] != report["edge_feature_dim"]
                or data.random_walk_pe.shape[1] != report["rwse_dim"]
            ):
                raise RuntimeError(f"Graph feature dimensions failed: {path}")
            if feature_schema == "ogb":
                if data.x.dtype != torch.long or data.edge_attr.dtype != torch.long:
                    raise RuntimeError(f"OGB categorical dtype failed: {path}")
                for column, categories in enumerate(ATOM_FEATURE_DIMS):
                    values = data.x[:, column]
                    if values.numel() and (
                        int(values.min()) < 0 or int(values.max()) >= categories
                    ):
                        raise RuntimeError(f"OGB atom category range failed: {path}")
                for column, categories in enumerate(BOND_FEATURE_DIMS):
                    values = data.edge_attr[:, column]
                    if values.numel() and (
                        int(values.min()) < 0 or int(values.max()) >= categories
                    ):
                        raise RuntimeError(f"OGB bond category range failed: {path}")
            if not all(torch.isfinite(value).all() for value in (data.x, data.edge_attr, data.random_walk_pe, data.y)):
                raise RuntimeError(f"Graph contains non-finite tensors: {path}")
            counts[item["role"]] += len(dataset)
            if item["role"] == "train":
                label_sum += float(targets.sum())
                label_squared_sum += float(np.square(targets).sum())
            del dataset, data
        reports.append({
            "shard_index": shard_index,
            "report_path": report_path.relative_to(graph_dir).as_posix(),
            "report_sha256": sha256_file(report_path),
        })
        gc.collect()

    expected_counts = {
        "train": int(len(splits["train"])),
        "valid": int(len(splits["valid"])),
    }
    if failures:
        raise RuntimeError(f"Official graph construction had {len(failures)} failures")
    if counts != expected_counts:
        raise RuntimeError(f"Official graph counts changed: {counts} != {expected_counts}")
    expected_mask = np.zeros(len(lookup), dtype=np.bool_)
    expected_mask[splits["train"]] = True
    expected_mask[splits["valid"]] = True
    if not np.array_equal(seen, expected_mask):
        raise RuntimeError("Accepted graph source_idx coverage is incomplete")
    mean = label_sum / counts["train"]
    std = math.sqrt(max(label_squared_sum / counts["train"] - mean * mean, 1.0e-12))
    first_report = json.loads(
        (graph_dir / "reports" / "shard_0000.json").read_text(encoding="utf-8")
    )
    acceptance = {
        "format": "molgap-pcqm4mv2-official-edge-state-graph-acceptance-v2",
        "status": "accepted",
        "archive_sha256": source_manifest["archive_sha256"],
        "graph_builder_version": builder_version,
        "feature_schema": feature_schema,
        "source_manifest_sha256": sha256_file(rows_dir / "manifest.json"),
        "counts": counts,
        "atom_list": first_report["atom_list"],
        "node_feature_dim": first_report["node_feature_dim"],
        "edge_feature_dim": first_report["edge_feature_dim"],
        "atom_feature_dims": first_report.get("atom_feature_dims"),
        "bond_feature_dims": first_report.get("bond_feature_dims"),
        "rwse_dim": first_report["rwse_dim"],
        "target_mean_gap": mean,
        "target_std_gap": std,
        "failures": 0,
        "unsanitized_fallback_rows": len(unsanitized_fallback),
        "unsanitized_fallback_source_idx": sorted(unsanitized_fallback),
        "test_graphs_built": False,
        "external_data_used": False,
        "reports": reports,
    }
    atomic_json(output_path, acceptance)
    return acceptance


def _graph_files(graph_dir: Path, role: str) -> list[Path]:
    paths = sorted((Path(graph_dir) / role).glob(f"{role}_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {role} graph shards in {graph_dir}")
    return paths


def _make_model(config: OfficialEdgeStateConfig, in_channels: int) -> nn.Module:
    if config.feature_schema == "ogb":
        if in_channels != len(ATOM_FEATURE_DIMS):
            raise ValueError("OGB atom feature width changed")
        return CategoricalEdgeStateStructuralGPSWrapper(
            atom_feature_dims=ATOM_FEATURE_DIMS,
            bond_feature_dims=BOND_FEATURE_DIMS,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            n_targets=1,
            rwse_dim=config.rwse_dim,
            edge_state_channels=config.edge_state_channels,
        )
    if config.feature_schema != "legacy":
        raise ValueError(f"unsupported feature schema: {config.feature_schema}")
    return EdgeStateStructuralGPSWrapper(
        in_channels=in_channels,
        edge_dim=4,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        n_targets=1,
        rwse_dim=config.rwse_dim,
        edge_state_channels=config.edge_state_channels,
    )


def _official_warmup_cosine_factor(
    epoch: int,
    *,
    max_epochs: int,
    warmup_epochs: int,
    minimum_factor: float,
) -> float:
    if not 0 <= warmup_epochs < max_epochs:
        raise ValueError("warmup_epochs must be in [0, max_epochs)")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    cosine_epochs = max(max_epochs - warmup_epochs - 1, 1)
    progress = min(max(epoch - warmup_epochs, 0), cosine_epochs) / cosine_epochs
    return minimum_factor + (1.0 - minimum_factor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _official_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OfficialEdgeStateConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler == "cosine":
        if config.warmup_epochs:
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
            lr_lambda=lambda epoch: _official_warmup_cosine_factor(
                epoch,
                max_epochs=config.max_epochs,
                warmup_epochs=config.warmup_epochs,
                minimum_factor=minimum_factor,
            ),
        )
    raise ValueError(f"unsupported scheduler: {config.scheduler}")


def _forward(model: nn.Module, batch) -> torch.Tensor:
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


@torch.no_grad()
def evaluate_valid(
    model: nn.Module,
    graph_dir: Path,
    device: torch.device,
    batch_size: int,
    target_mean: float,
    target_std: float,
    *,
    return_predictions: bool = False,
):
    model.eval()
    absolute, count = 0.0, 0
    indices, predictions, targets = [], [], []
    for path in _graph_files(graph_dir, "valid"):
        dataset = PackedGraphDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = _forward(model, batch)
            prediction = prediction.float() * target_std + target_mean
            target = batch.y.view(-1)
            absolute += float((prediction - target).abs().sum())
            count += int(target.numel())
            if return_predictions:
                indices.append(batch.source_idx.view(-1).cpu())
                predictions.append(prediction.cpu())
                targets.append(target.cpu())
        del loader, dataset
    mae = absolute / max(count, 1)
    if not return_predictions:
        return mae
    order = torch.argsort(torch.cat(indices))
    return mae, {
        "source_idx": torch.cat(indices)[order],
        "prediction_eV": torch.cat(predictions)[order],
        "target_eV": torch.cat(targets)[order],
    }


def train_official_edge_state(
    graph_dir: Path,
    acceptance_path: Path,
    output_dir: Path,
    *,
    config: OfficialEdgeStateConfig = OfficialEdgeStateConfig(),
) -> dict:
    """Train from scratch on official train and select only on official valid."""
    started = time.monotonic()
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Official PCQM EdgeState training requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    acceptance_path = Path(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted" or acceptance.get("external_data_used"):
        raise RuntimeError("Official graph acceptance is missing or contaminated")
    if acceptance.get("feature_schema", "legacy") != config.feature_schema:
        raise RuntimeError("Official graph and model feature schemas differ")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = float(acceptance["target_mean_gap"])
    std = float(acceptance["target_std_gap"])
    model = _make_model(config, int(acceptance["node_feature_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _official_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.L1Loss()
    best_mae, best_epoch, wait, start_epoch = float("inf"), -1, 0, 0
    log = []
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    contract = asdict(config)
    acceptance_hash = sha256_file(acceptance_path)
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") != contract or state.get("acceptance_sha256") != acceptance_hash:
            raise RuntimeError("Official EdgeState resume contract changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_mae = float(state["best_valid_gap_mae_eV"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])
        start_epoch = int(state["epoch"]) + 1
        log = list(state["log"])

    base_train_paths = _graph_files(graph_dir, "train")
    for epoch in range(start_epoch, config.max_epochs):
        if time.monotonic() - started + (log[-1]["elapsed_s"] if log else 0.0) > config.hard_job_budget_s:
            print("hard job budget reached; checkpoint is resumable", flush=True)
            break
        epoch_started = time.monotonic()
        model.train()
        train_paths = list(base_train_paths)
        random.Random(config.seed + epoch).shuffle(train_paths)
        loss_sum, row_count = 0.0, 0
        for shard_number, path in enumerate(train_paths, start=1):
            dataset = PackedGraphDataset(path)
            generator = torch.Generator().manual_seed(
                config.seed * 1_000_000 + epoch * 10_000 + shard_number
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                pin_memory=True,
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    prediction = _forward(model, batch)
                    target = (batch.y.view(-1) - mean) / std
                    loss = criterion(prediction, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * batch.num_graphs
                row_count += int(batch.num_graphs)
            del loader, dataset
            gc.collect()
        scheduler.step()
        valid_mae = float(
            evaluate_valid(model, graph_dir, device, config.eval_batch_size, mean, std)
        )
        elapsed = time.monotonic() - epoch_started
        projection_epochs = config.projection_epochs or config.max_epochs
        projected = elapsed * projection_epochs
        if epoch == 0 and projected > config.max_projected_training_s:
            atomic_json(output_dir / "timing_gate.json", {
                "status": "rejected",
                "epoch_s": elapsed,
                "projected_training_s": projected,
                "limit_s": config.max_projected_training_s,
            })
            raise RuntimeError("Measured full training exceeds the declared timing gate")
        if epoch == 0:
            atomic_json(output_dir / "timing_gate.json", {
                "status": "accepted",
                "epoch_s": elapsed,
                "projection_epochs": projection_epochs,
                "projected_training_s": projected,
                "limit_s": config.max_projected_training_s,
            })
        improved = np.isfinite(valid_mae) and valid_mae < best_mae
        if improved:
            best_mae, best_epoch, wait = valid_mae, epoch, 0
            atomic_torch(best_path, {
                "format": "molgap-pcqm4mv2-official-edge-state-best-v1",
                "config": contract,
                "model": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch,
                "best_valid_gap_mae_eV": best_mae,
                "target_mean_gap": mean,
                "target_std_gap": std,
                "acceptance_sha256": acceptance_hash,
                "external_data_used": False,
            })
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_gap_l1_normalized": loss_sum / max(row_count, 1),
            "train_rows": row_count,
            "valid_gap_mae_eV": valid_mae,
            "elapsed_s": elapsed,
            "projected_training_s": projected,
            "selected": bool(improved),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        log.append(row)
        atomic_torch(last_path, {
            "format": "molgap-pcqm4mv2-official-edge-state-checkpoint-v1",
            "config": contract,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "wait": wait,
            "log": log,
            "target_mean_gap": mean,
            "target_std_gap": std,
            "acceptance_sha256": acceptance_hash,
        })
        atomic_json(output_dir / "progress.json", {
            "status": "training",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "elapsed_s": time.monotonic() - started,
            "resumable_checkpoint": str(last_path),
        })
        print(
            f"official-edge-state ep{epoch:02d} "
            f"train={row['train_gap_l1_normalized']:.6f} "
            f"valid={valid_mae:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break

    if not best_path.is_file():
        raise RuntimeError("Official EdgeState training has no finite selected model")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    valid_mae, predictions = evaluate_valid(
        model,
        graph_dir,
        device,
        config.eval_batch_size,
        mean,
        std,
        return_predictions=True,
    )
    atomic_torch(output_dir / "valid_predictions.pt", predictions)
    metrics = {
        "format": "molgap-pcqm4mv2-official-edge-state-training-v1",
        "status": "complete",
        "config": contract,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best["best_epoch"]),
        "valid_gap_mae_eV": float(valid_mae),
        "train_log": log,
        "official_train_rows": acceptance["counts"]["train"],
        "official_valid_rows": acceptance["counts"]["valid"],
        "official_valid_used": True,
        "official_test_used": False,
        "external_data_used": False,
        "pretrained_weights_used": False,
        "production_registry_changed": False,
        "runtime_s": time.monotonic() - started,
        "best_sha256": sha256_file(best_path),
        "valid_predictions_sha256": sha256_file(output_dir / "valid_predictions.pt"),
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(output_dir / "completion_manifest.json", {
        "status": "complete",
        "best": {"path": "best.pt", "sha256": metrics["best_sha256"]},
        "metrics": {"path": "metrics.json", "sha256": sha256_file(output_dir / "metrics.json")},
        "valid_predictions": {
            "path": "valid_predictions.pt",
            "sha256": metrics["valid_predictions_sha256"],
        },
    })
    return metrics


def continue_official_edge_state(
    graph_dir: Path,
    acceptance_path: Path,
    source_dir: Path,
    output_dir: Path,
    *,
    config: OfficialEdgeStateContinuationConfig = OfficialEdgeStateContinuationConfig(),
) -> dict:
    """Continue a completed run in an isolated directory with a new schedule.

    The source run is immutable.  The continuation keeps its model and Adam
    state, resets only the learning-rate schedule, and records the source
    hashes plus the continuation contract in every checkpoint.
    """
    if config.additional_epochs < 1:
        raise ValueError("additional_epochs must be positive")
    if config.scheduler == "warmup_cosine" and not (
        0 <= config.warmup_epochs < config.additional_epochs
    ):
        raise ValueError("warmup_epochs must be in [0, additional_epochs)")
    if config.scheduler == "cosine" and config.warmup_epochs:
        raise ValueError("cosine scheduler does not accept warmup_epochs")

    started = time.monotonic()
    graph_dir, acceptance_path = Path(graph_dir), Path(acceptance_path)
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "accepted" or acceptance.get("external_data_used"):
        raise RuntimeError("Official graph acceptance is missing or contaminated")
    source_last_path = source_dir / "last.pt"
    source_best_path = source_dir / "best.pt"
    source_completion_path = source_dir / "completion_manifest.json"
    for path in (source_last_path, source_best_path, source_completion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    acceptance_hash = sha256_file(acceptance_path)
    source_last_hash = sha256_file(source_last_path)
    source_best_hash = sha256_file(source_best_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete":
            metrics_path = output_dir / "metrics.json"
            if not metrics_path.is_file():
                raise RuntimeError("complete continuation is missing metrics")
            return json.loads(metrics_path.read_text(encoding="utf-8"))

    source_state = torch.load(source_last_path, map_location="cpu", weights_only=False)
    source_best = torch.load(source_best_path, map_location="cpu", weights_only=False)
    base_config = OfficialEdgeStateConfig(**source_state["config"])
    if source_state.get("acceptance_sha256") != acceptance_hash:
        raise RuntimeError("source checkpoint does not match accepted graph cache")
    if source_best.get("config") != source_state.get("config"):
        raise RuntimeError("source best and last checkpoints use different configs")
    if source_best.get("acceptance_sha256") != acceptance_hash:
        raise RuntimeError("source best checkpoint does not match graph acceptance")
    if acceptance.get("feature_schema", "legacy") != base_config.feature_schema:
        raise RuntimeError("source model and graph feature schemas differ")
    source_epoch = int(source_state["epoch"])
    if source_epoch < 0:
        raise RuntimeError("source checkpoint has no completed epoch")
    in_channels = int(acceptance["node_feature_dim"])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Official PCQM EdgeState continuation requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(base_config.seed)

    continuation_contract = {
        "format": "molgap-pcqm4mv2-official-edge-state-continuation-v1",
        "base_config": asdict(base_config),
        "continuation_config": asdict(config),
        "source_last_sha256": source_last_hash,
        "source_best_sha256": source_best_hash,
        "acceptance_sha256": acceptance_hash,
        "source_epoch": source_epoch,
    }
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    continuation_schedule = OfficialEdgeStateConfig(
        learning_rate=config.learning_rate,
        max_epochs=config.additional_epochs,
        scheduler=config.scheduler,
        warmup_epochs=config.warmup_epochs,
        minimum_learning_rate=config.minimum_learning_rate,
    )
    model = _make_model(base_config, in_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=base_config.weight_decay,
    )
    scheduler = _official_scheduler(optimizer, continuation_schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.L1Loss()
    mean = float(acceptance["target_mean_gap"])
    std = float(acceptance["target_std_gap"])
    log = []
    elapsed_offset = 0.0
    wait = 0
    best_mae = float(source_best["best_valid_gap_mae_eV"])
    best_epoch = int(source_best["best_epoch"])
    start_local_epoch = 0
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("continuation_contract") != continuation_contract:
            raise RuntimeError("continuation checkpoint contract changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["initial_lr"] = config.learning_rate
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_mae = float(state["best_valid_gap_mae_eV"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])
        start_local_epoch = int(state["local_epoch"]) + 1
        log = list(state["log"])
        elapsed_offset = float(state.get("continuation_elapsed_s", 0.0))
    else:
        model.load_state_dict(source_state["model"], strict=True)
        optimizer.load_state_dict(source_state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["initial_lr"] = config.learning_rate
        scaler.load_state_dict(source_state["scaler"])
        atomic_torch(best_path, {
            "format": "molgap-pcqm4mv2-official-edge-state-continuation-best-v1",
            "continuation_contract": continuation_contract,
            "model": copy.deepcopy(source_best["model"]),
            "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "target_mean_gap": mean,
            "target_std_gap": std,
            "source_best_sha256": source_best_hash,
            "acceptance_sha256": acceptance_hash,
        })

    base_train_paths = _graph_files(graph_dir, "train")
    stopped_for_budget = False
    for local_epoch in range(start_local_epoch, config.additional_epochs):
        if elapsed_offset + time.monotonic() - started > config.hard_job_budget_s:
            stopped_for_budget = True
            print("continuation hard job budget reached; checkpoint is resumable", flush=True)
            break
        epoch_started = time.monotonic()
        global_epoch = source_epoch + 1 + local_epoch
        model.train()
        train_paths = list(base_train_paths)
        random.Random(base_config.seed + global_epoch).shuffle(train_paths)
        loss_sum, row_count = 0.0, 0
        for shard_number, path in enumerate(train_paths, start=1):
            dataset = PackedGraphDataset(path)
            generator = torch.Generator().manual_seed(
                base_config.seed * 1_000_000 + global_epoch * 10_000 + shard_number
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                pin_memory=True,
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    prediction = _forward(model, batch)
                    target = (batch.y.view(-1) - mean) / std
                    loss = criterion(prediction, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * batch.num_graphs
                row_count += int(batch.num_graphs)
            del loader, dataset
            gc.collect()
        scheduler.step()
        valid_mae = float(
            evaluate_valid(
                model,
                graph_dir,
                device,
                config.eval_batch_size,
                mean,
                std,
            )
        )
        elapsed = time.monotonic() - epoch_started
        improved = np.isfinite(valid_mae) and valid_mae < best_mae
        if improved:
            best_mae, best_epoch, wait = valid_mae, global_epoch, 0
            atomic_torch(best_path, {
                "format": "molgap-pcqm4mv2-official-edge-state-continuation-best-v1",
                "continuation_contract": continuation_contract,
                "model": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch,
                "best_valid_gap_mae_eV": best_mae,
                "target_mean_gap": mean,
                "target_std_gap": std,
                "source_best_sha256": source_best_hash,
                "acceptance_sha256": acceptance_hash,
            })
        else:
            wait += 1
        row = {
            "local_epoch": local_epoch,
            "epoch": global_epoch,
            "train_gap_l1_normalized": loss_sum / max(row_count, 1),
            "train_rows": row_count,
            "valid_gap_mae_eV": valid_mae,
            "elapsed_s": elapsed,
            "selected": bool(improved),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        log.append(row)
        continuation_elapsed = elapsed_offset + time.monotonic() - started
        atomic_torch(last_path, {
            "format": "molgap-pcqm4mv2-official-edge-state-continuation-checkpoint-v1",
            "continuation_contract": continuation_contract,
            "local_epoch": local_epoch,
            "epoch": global_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "wait": wait,
            "log": log,
            "continuation_elapsed_s": continuation_elapsed,
            "target_mean_gap": mean,
            "target_std_gap": std,
            "source_last_sha256": source_last_hash,
            "source_best_sha256": source_best_hash,
            "acceptance_sha256": acceptance_hash,
        })
        atomic_json(output_dir / "progress.json", {
            "status": "training",
            "local_epoch": local_epoch,
            "epoch": global_epoch,
            "best_epoch": best_epoch,
            "best_valid_gap_mae_eV": best_mae,
            "continuation_elapsed_s": continuation_elapsed,
            "resumable_checkpoint": str(last_path),
        })
        print(
            f"official-edge-state-cont ep{global_epoch:02d} "
            f"train={row['train_gap_l1_normalized']:.6f} "
            f"valid={valid_mae:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break

    if not last_path.is_file():
        raise RuntimeError("continuation produced no resumable checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    valid_mae, predictions = evaluate_valid(
        model,
        graph_dir,
        device,
        config.eval_batch_size,
        mean,
        std,
        return_predictions=True,
    )
    atomic_torch(output_dir / "valid_predictions.pt", predictions)
    completed_epochs = len(log)
    schedule_complete = (
        not stopped_for_budget
        and (start_local_epoch + completed_epochs >= config.additional_epochs or wait >= config.patience)
    )
    status = "complete" if schedule_complete else "partial"
    metrics = {
        "format": "molgap-pcqm4mv2-official-edge-state-continuation-v1",
        "status": status,
        "continuation_contract": continuation_contract,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_best_epoch": int(source_best["best_epoch"]),
        "best_epoch": int(best["best_epoch"]),
        "source_best_valid_gap_mae_eV": float(source_best["best_valid_gap_mae_eV"]),
        "valid_gap_mae_eV": float(valid_mae),
        "delta_vs_source_best_eV": float(valid_mae - float(source_best["best_valid_gap_mae_eV"])),
        "train_log": log,
        "official_train_rows": acceptance["counts"]["train"],
        "official_valid_rows": acceptance["counts"]["valid"],
        "official_valid_used": True,
        "official_test_used": False,
        "external_data_used": False,
        "pretrained_weights_used": False,
        "warm_started_from_completed_run": True,
        "production_registry_changed": False,
        "runtime_s": elapsed_offset + time.monotonic() - started,
        "completed_additional_epochs": completed_epochs,
        "best_sha256": sha256_file(best_path),
        "valid_predictions_sha256": sha256_file(output_dir / "valid_predictions.pt"),
    }
    atomic_json(output_dir / "metrics.json", metrics)
    atomic_json(completion_path, {
        "status": status,
        "source": {
            "last": {"path": str(source_last_path), "sha256": source_last_hash},
            "best": {"path": str(source_best_path), "sha256": source_best_hash},
        },
        "best": {"path": "best.pt", "sha256": metrics["best_sha256"]},
        "last": {"path": "last.pt", "sha256": sha256_file(last_path)},
        "metrics": {"path": "metrics.json", "sha256": sha256_file(output_dir / "metrics.json")},
        "valid_predictions": {
            "path": "valid_predictions.pt",
            "sha256": metrics["valid_predictions_sha256"],
        },
    })
    return metrics


def _iter_official_rows(archive: Path) -> Iterable[tuple[int, str, float]]:
    bundle, compressed, stream = _open_official_csv(archive)
    text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
    try:
        reader = csv.DictReader(text)
        if reader.fieldnames != ["idx", "smiles", "homolumogap"]:
            raise ValueError(f"Unexpected official CSV columns: {reader.fieldnames}")
        for row in reader:
            raw = row["homolumogap"]
            yield int(row["idx"]), row["smiles"], float(raw) if raw else float("nan")
    finally:
        text.close()
        compressed.close()
        bundle.close()


def predict_official_tests_from_raw(
    archive: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    workers: int = 6,
    part_rows: int = 20_000,
    inference_limit_s: float = 4.0 * 3600.0,
) -> dict:
    """Run the frozen model from raw test SMILES and write OGB NPZ files."""
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Official PCQM test inference requires a CUDA GPU")
    archive, checkpoint_path, output_dir = Path(archive), Path(checkpoint_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OfficialEdgeStateConfig(**checkpoint["config"])
    in_channels = (
        len(ATOM_FEATURE_DIMS)
        if config.feature_schema == "ogb"
        else len(OFFICIAL_ATOM_LIST) + 3
    )
    model = _make_model(config, in_channels)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    splits = load_official_splits(archive)
    test_code = np.full(validate_official_splits(splits)["rows"], -1, dtype=np.int8)
    test_code[splits["test-dev"]] = 2
    test_code[splits["test-challenge"]] = 3
    mean = float(checkpoint["target_mean_gap"])
    std = float(checkpoint["target_std_gap"])
    manifest_path = output_dir / "parts_manifest.json"
    resumed = manifest_path.is_file()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if resumed
        else {
            "format": "molgap-pcqm4mv2-official-test-parts-v1",
            "status": "predicting",
            "archive_sha256": sha256_file(archive),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parts": [],
        }
    )
    if manifest["archive_sha256"] != sha256_file(archive) or manifest["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise RuntimeError("Official test inference resume contract changed")
    completed = set()
    for item in manifest["parts"]:
        path = output_dir / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Official test prediction part changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        completed.update(payload["source_idx"].tolist())

    pending = []
    part_index = len(manifest["parts"])

    @torch.no_grad()
    def flush(rows: list[tuple[int, str, float, int]]) -> None:
        nonlocal part_index
        if not rows:
            return
        if workers > 1:
            context = mp.get_context("spawn" if os.name == "nt" else "fork")
            with context.Pool(
                processes=workers,
                initializer=_init_graph_worker,
                initargs=(OFFICIAL_ATOM_LIST, config.rwse_dim, config.feature_schema),
            ) as pool:
                built = list(pool.imap(_graph_from_row, rows, chunksize=128))
        else:
            _init_graph_worker(
                OFFICIAL_ATOM_LIST, config.rwse_dim, config.feature_schema
            )
            built = [_graph_from_row(row) for row in rows]
        failures = [failure for _, failure in built if failure is not None]
        if failures:
            raise RuntimeError(f"Official test preprocessing failed: {failures[:3]}")
        graphs = [graph for graph, _ in built]
        indices, predictions = [], []
        loader = DataLoader(graphs, batch_size=config.eval_batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = _forward(model, batch)
            indices.append(batch.source_idx.view(-1).cpu())
            predictions.append((prediction.float() * std + mean).cpu())
        payload = {
            "source_idx": torch.cat(indices),
            "prediction_eV": torch.cat(predictions),
        }
        path = output_dir / "parts" / f"test_predictions_part_{part_index:03d}.pt"
        atomic_torch(path, payload)
        manifest["parts"].append({
            "path": path.relative_to(output_dir).as_posix(),
            "rows": len(payload["source_idx"]),
            "sha256": sha256_file(path),
        })
        atomic_json(manifest_path, manifest)
        part_index += 1
        print(f"official test part {part_index}: {len(rows):,} rows", flush=True)

    for source_idx, smiles, _ in _iter_official_rows(archive):
        code = int(test_code[source_idx])
        if code < 2 or source_idx in completed:
            continue
        pending.append((source_idx, smiles, float("nan"), code))
        if len(pending) >= part_rows:
            flush(pending)
            pending = []
    flush(pending)

    all_indices, all_predictions = [], []
    for item in manifest["parts"]:
        payload = torch.load(output_dir / item["path"], map_location="cpu", weights_only=False)
        all_indices.append(payload["source_idx"].numpy())
        all_predictions.append(payload["prediction_eV"].numpy())
    source_idx = np.concatenate(all_indices).astype(np.int64)
    prediction = np.concatenate(all_predictions).astype(np.float32)
    if len(np.unique(source_idx)) != len(source_idx):
        raise RuntimeError("Official test predictions contain duplicate source_idx")
    lookup = {int(index): float(value) for index, value in zip(source_idx, prediction)}
    outputs = {}
    for name in ("test-dev", "test-challenge"):
        expected = splits[name]
        if any(int(index) not in lookup for index in expected):
            raise RuntimeError(f"Official {name} predictions are incomplete")
        values = np.asarray([lookup[int(index)] for index in expected], dtype=np.float32)
        path = atomic_ogb_submission(output_dir, name, values)
        outputs[name] = {
            "path": path.name,
            "rows": len(values),
            "sha256": sha256_file(path),
        }
    elapsed = time.monotonic() - started
    acceptance = validate_submission_files(output_dir)
    result = {
        "format": "molgap-pcqm4mv2-official-test-submission-v1",
        "status": "complete",
        "outputs": outputs,
        "raw_smiles_preprocessing_included": True,
        "single_gpu": True,
        "cpu_workers": workers,
        "resumed": resumed,
        "elapsed_s": elapsed,
        "inference_limit_s": inference_limit_s,
        "official_4h_timing_pass": bool(not resumed and elapsed <= inference_limit_s),
        "external_data_used": False,
        "ogb_acceptance": acceptance,
    }
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    atomic_json(output_dir / "submission_manifest.json", result)
    if not resumed and elapsed > inference_limit_s:
        raise RuntimeError("Raw-SMILES official test inference exceeded four hours")
    return result
