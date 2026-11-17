"""PCQM-only data contracts for the Route B precision architecture."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .pcqm_expert import (
    PackedGraphDataset,
    _save_packed_graphs,
    atomic_json,
    load_expanded_rows,
    sha256_file,
)
from .utils import compute_gasteiger_charges

ATOM_LIST = (6, 7, 8, 9, 16, 17, 15, 35, 14, 5, 34, 32, 33, 12, 2)
ROLE_NAMES = {0: "train", 1: "dev", 2: "official"}
DEFAULT_SHARD_ROWS = 5_000


def _row_seed(base_seed: int, source_idx: int) -> int:
    return int((base_seed * 1_000_003 + source_idx) % 2_147_483_647)


def _gps_graph(molecule: Chem.Mol) -> Data | None:
    node_features = []
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        node_features.append(
            [float(atomic_number == value) for value in ATOM_LIST]
            + [
                atom.GetDegree() / 4.0,
                atom.GetFormalCharge() / 2.0,
                float(atom.GetIsAromatic()),
            ]
        )
    if not node_features:
        return None
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    rows, columns, edge_features = [], [], []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_types.get(bond.GetBondType(), 0)
        feature = [float(bond_type == index) for index in range(4)]
        rows.extend((left, right))
        columns.extend((right, left))
        edge_features.extend((feature, feature))
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor([rows, columns], dtype=torch.long),
        edge_attr=(
            torch.tensor(edge_features, dtype=torch.float32)
            if edge_features
            else torch.zeros((0, 4), dtype=torch.float32)
        ),
    )


def _conformer_graph(
    molecule: Chem.Mol,
    source_idx: int,
    base_seed: int,
) -> Data | None:
    molecule = AllChem.AddHs(Chem.Mol(molecule))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = _row_seed(base_seed, source_idx)
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
    except Exception:
        pass
    conformer = molecule.GetConformer()
    return Data(
        z=torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            dtype=torch.long,
        ),
        pos=torch.tensor(conformer.GetPositions(), dtype=torch.float32),
        charges=torch.tensor(
            compute_gasteiger_charges(molecule),
            dtype=torch.float32,
        ),
    )


def build_route_b_row(
    row: tuple[int, str, float, int],
) -> tuple[int, tuple[Data, Data, Data] | None]:
    """Build aligned expanded-2D, primary-3D, and secondary-3D graphs."""
    source_idx, smiles, gap, split_code = row
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or split_code not in ROLE_NAMES:
        return source_idx, None
    gps = _gps_graph(molecule)
    primary = _conformer_graph(molecule, source_idx, 42)
    secondary = _conformer_graph(molecule, source_idx, 43)
    if gps is None or primary is None or secondary is None:
        return source_idx, None
    for graph in (gps, primary, secondary):
        graph.y = torch.tensor([gap], dtype=torch.float32)
        graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
        graph.split_code = torch.tensor([split_code], dtype=torch.int8)
    return source_idx, (gps, primary, secondary)


def _split_lookup(gine_cache: Path, maximum_idx: int) -> np.ndarray:
    lookup = np.full(maximum_idx + 1, -1, dtype=np.int8)
    for split_code, role in ROLE_NAMES.items():
        for path in sorted(gine_cache.glob(f"{role}_shard_*.pt")):
            dataset = PackedGraphDataset(path)
            indices = dataset._data.sample_idx.view(-1).numpy()
            lookup[indices] = split_code
    return lookup


def _ordered_rows(rows: pd.DataFrame, lookup: np.ndarray) -> pd.DataFrame:
    selected = rows.copy()
    source_idx = selected["idx"].to_numpy(dtype=np.int64)
    selected["split_code"] = lookup[source_idx]
    train_pool = selected[selected["source_split"] == 0]
    official = selected[selected["source_split"] == 2]
    order = np.random.default_rng(42).permutation(len(train_pool))
    return pd.concat(
        (train_pool.iloc[order], official.sort_values("idx")),
        ignore_index=True,
    )


def build_route_b_cache(
    *,
    raw_csv: Path,
    accepted_valid_predictions: Path,
    gine_cache: Path,
    cache_dir: Path,
    total_train_rows: int = 1_000_000,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    workers: int = 12,
) -> dict:
    """Build resumable aligned Route B graph shards for PCQM-only training."""
    rows, input_contract = load_expanded_rows(
        raw_csv,
        accepted_valid_predictions,
        total_train_rows=total_train_rows,
    )
    lookup = _split_lookup(gine_cache, int(rows["idx"].max()))
    rows = _ordered_rows(rows, lookup)
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(cache_dir / "input_contract.json", input_contract)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["source_rows"] != len(rows)
            or manifest["shard_rows"] != shard_rows
            or manifest["atom_list"] != list(ATOM_LIST)
        ):
            raise RuntimeError("PCQM Route B cache contract changed")
        if manifest["status"] == "complete":
            return manifest
    else:
        manifest = {
            "format": "molgap-pcqm-route-b-aligned-cache-v1",
            "status": "building",
            "source_rows": len(rows),
            "processed_rows": 0,
            "accepted_rows": 0,
            "failed_rows": 0,
            "failure_source_idx": [],
            "shard_rows": shard_rows,
            "atom_list": list(ATOM_LIST),
            "node_feature_dim": len(ATOM_LIST) + 3,
            "conformer_contract": {
                "method": "ETKDGv3+MMFF200",
                "primary_seed": 42,
                "secondary_seed": 43,
            },
            "split_counts": {"train": 0, "dev": 0, "official": 0},
            "shards": [],
        }

    start = int(manifest["processed_rows"])
    if start != len(manifest["shards"]) * shard_rows:
        raise RuntimeError("PCQM Route B resume boundary changed")
    modalities = ("gps", "primary", "secondary")
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for begin in range(start, len(rows), shard_rows):
            started = time.perf_counter()
            end = min(begin + shard_rows, len(rows))
            work = [
                (
                    int(row.idx),
                    str(row.smiles),
                    float(row.homolumogap),
                    int(row.split_code),
                )
                for row in rows.iloc[begin:end].itertuples(index=False)
            ]
            grouped = {
                modality: {role: [] for role in ROLE_NAMES.values()}
                for modality in modalities
            }
            failures = []
            for source_idx, result in pool.imap(
                build_route_b_row,
                work,
                chunksize=25,
            ):
                if result is None:
                    failures.append(source_idx)
                    continue
                role = ROLE_NAMES[int(result[0].split_code.item())]
                for modality, graph in zip(modalities, result):
                    grouped[modality][role].append(graph)

            files = {}
            counts = {role: len(grouped["gps"][role]) for role in ROLE_NAMES.values()}
            shard_id = begin // shard_rows
            for modality in modalities:
                modality_dir = cache_dir / modality
                modality_dir.mkdir(parents=True, exist_ok=True)
                for role, graphs in grouped[modality].items():
                    if not graphs:
                        continue
                    path = modality_dir / f"{role}_shard_{shard_id:03d}.pt"
                    _save_packed_graphs(path, graphs)
                    files[path.relative_to(cache_dir).as_posix()] = sha256_file(path)

            manifest["processed_rows"] = end
            manifest["accepted_rows"] += sum(counts.values())
            manifest["failed_rows"] += len(failures)
            manifest["failure_source_idx"].extend(failures)
            for role, count in counts.items():
                manifest["split_counts"][role] += count
            manifest["shards"].append(
                {
                    "begin": begin,
                    "end": end,
                    "counts": counts,
                    "files": files,
                }
            )
            atomic_json(manifest_path, manifest)
            print(
                f"route-b shard {shard_id:03d}: accepted={sum(counts.values())} "
                f"failed={len(failures)} {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if manifest["accepted_rows"] + manifest["failed_rows"] != len(rows):
        raise RuntimeError("PCQM Route B cache rows do not reconcile")
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    return manifest
