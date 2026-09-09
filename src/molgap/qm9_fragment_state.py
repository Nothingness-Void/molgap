"""Deterministic fragment-state GPS screen for the Track C QM9 funnel.

The experiment keeps the transferable OGB atom/bond graph, RWSE16 input, and
EdgeState GPS9 backbone unchanged.  At layers 3, 6, and 9 a BRICS-defined
fragment stream is updated from the atom states and broadcast back through a
zero-start low-rank residual.  The cache contains only train and validation
roles; no QM9 test role or official PCQM role is materialized.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from .pcqm_gap_architecture import (
    OGBEdgeStateStructuralGPSWrapper,
    _LowRankGatedProjection,
)


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
FRAGMENT_BLOCKS = (3, 6, 9)
FRAGMENT_FEATURE_DIM = 8
FRAGMENT_EDGE_DIM = 4
FRAGMENT_CHANNELS = 32
FRAGMENT_EXCHANGE_RANK = 16
SCREEN_FORMAT = "molgap-qm9-fragment-state-screen-v1"
CACHE_FORMAT = "molgap-qm9-fragment-state-cache-v1"
ACCEPTANCE_FORMAT = "molgap-qm9-fragment-state-cache-acceptance-v1"
FRAGMENT_ALGORITHM = "brics-heavy-atom-components-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FragmentStateData(Data):
    """PyG graph with explicit offsets for the fragment-level stream."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"fragment_id", "fragment_edge_index"}:
            return int(self.fragment_features.shape[0])
        return super().__inc__(key, value, *args, **kwargs)


def make_fragment_data_class():
    """Return the module-level PyG data class used by the fragment cache."""
    return FragmentStateData


def _record_molecule(record: dict, supplier):
    from rdkit import Chem

    raw_index = int(str(record["name"]).split("_")[-1]) - 1
    molecule = supplier[raw_index]
    if molecule is None:
        raise ValueError("QM9 SDF molecule is missing")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("canonical QM9 SMILES did not sanitize")
    return molecule, smiles


def _canonical_valid_pool(records: list[dict], supplier):
    valid = []
    invalid = []
    for source_idx, record in enumerate(records):
        try:
            _record_molecule(record, supplier)
        except Exception as error:
            invalid.append(
                {
                    "source_idx": int(source_idx),
                    "name": str(record["name"]),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        else:
            valid.append(int(source_idx))
    return np.asarray(valid, dtype=np.int64), invalid


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _fixed_split_from_pool(pool: np.ndarray):
    requested = TRAIN_ROWS + VALIDATION_ROWS
    if len(pool) < requested:
        raise ValueError(f"QM9 valid pool has {len(pool)} rows, need {requested}")
    order = np.random.RandomState(SPLIT_SEED).permutation(len(pool))[:requested]
    return pool[order[:TRAIN_ROWS]], pool[order[TRAIN_ROWS:]]


def _brics_cuts(molecule):
    from rdkit.Chem import BRICS

    cuts = set()
    for item in BRICS.FindBRICSBonds(molecule):
        endpoints = item[0]
        left, right = sorted((int(endpoints[0]), int(endpoints[1])))
        if molecule.GetBondBetweenAtoms(left, right) is not None:
            cuts.add((left, right))
    return tuple(sorted(cuts))


def _fragment_edge_feature(bond) -> list[float]:
    value = float(bond.GetBondTypeAsDouble())
    return [
        float(value == 1.0),
        float(value == 2.0),
        float(value == 3.0),
        float(value == 1.5),
    ]


def _fragment_payload(molecule):
    """Return atom fragment IDs, fragment features, and directed cut edges."""
    atom_count = molecule.GetNumAtoms()
    cuts = set(_brics_cuts(molecule))
    parent = list(range(atom_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for bond in molecule.GetBonds():
        endpoints = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if endpoints not in cuts:
            union(*endpoints)

    components: dict[int, list[int]] = {}
    for atom_index in range(atom_count):
        components.setdefault(find(atom_index), []).append(atom_index)
    ordered = sorted(components.values(), key=lambda atoms: min(atoms))
    atom_to_fragment = {}
    for fragment_index, atoms in enumerate(ordered):
        for atom_index in atoms:
            atom_to_fragment[atom_index] = fragment_index

    cut_degree = [0] * len(ordered)
    for left, right in cuts:
        cut_degree[atom_to_fragment[left]] += 1
        cut_degree[atom_to_fragment[right]] += 1

    features = []
    for fragment_index, atoms in enumerate(ordered):
        atom_objects = [molecule.GetAtomWithIdx(index) for index in atoms]
        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in atom_objects], dtype=np.float32
        )
        degrees = np.asarray(
            [atom.GetTotalDegree() for atom in atom_objects], dtype=np.float32
        )
        hetero = np.asarray(
            [atom.GetAtomicNum() != 6 for atom in atom_objects], dtype=np.float32
        )
        aromatic = np.asarray(
            [atom.GetIsAromatic() for atom in atom_objects], dtype=np.float32
        )
        in_ring = np.asarray(
            [atom.IsInRing() for atom in atom_objects], dtype=np.float32
        )
        size = float(len(atoms))
        features.append(
            [
                size / 16.0,
                float(atomic_numbers.mean()) / 10.0,
                float(atomic_numbers.max()) / 10.0,
                float(hetero.mean()),
                float(aromatic.mean()),
                float(in_ring.mean()),
                float(degrees.mean()) / 4.0,
                float(cut_degree[fragment_index]) / 4.0,
            ]
        )

    edge_indices = []
    edge_features = []
    for left, right in cuts:
        left_fragment = atom_to_fragment[left]
        right_fragment = atom_to_fragment[right]
        bond = molecule.GetBondBetweenAtoms(left, right)
        if bond is None:
            raise RuntimeError("BRICS cut does not map to a molecular bond")
        feature = _fragment_edge_feature(bond)
        edge_indices.extend(
            ((left_fragment, right_fragment), (right_fragment, left_fragment))
        )
        edge_features.extend((feature, feature))

    return (
        torch.tensor(
            [atom_to_fragment[index] for index in range(atom_count)],
            dtype=torch.long,
        ),
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        if edge_indices
        else torch.empty((2, 0), dtype=torch.long),
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features
        else torch.empty((0, FRAGMENT_EDGE_DIM), dtype=torch.float32),
    )


def _make_graph(record: dict, source_idx: int, supplier):
    from ogb.utils.mol import smiles2graph
    from torch_geometric.transforms import AddRandomWalkPE

    molecule, smiles = _record_molecule(record, supplier)
    payload = smiles2graph(smiles)
    fragment_id, fragment_features, fragment_edge_index, fragment_edge_attr = (
        _fragment_payload(molecule)
    )
    if payload["node_feat"].shape[0] != molecule.GetNumAtoms():
        raise RuntimeError("OGB atom and canonical molecule counts differ")
    graph = FragmentStateData(
        x=torch.as_tensor(payload["node_feat"], dtype=torch.long),
        edge_index=torch.as_tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(payload["edge_feat"], dtype=torch.long),
        y=record["y"].view(-1)[4].float().view(1),
        row_id=torch.tensor([int(source_idx)], dtype=torch.long),
        fragment_id=fragment_id,
        fragment_features=fragment_features,
        fragment_edge_index=fragment_edge_index,
        fragment_edge_attr=fragment_edge_attr,
        fragment_count=torch.tensor([fragment_features.shape[0]], dtype=torch.long),
    )
    if graph.x.ndim != 2 or graph.x.shape[1] != 9:
        raise RuntimeError("OGB atom representation changed")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
        raise RuntimeError("OGB bond representation changed")
    return AddRandomWalkPE(walk_length=RWSE_DIM, attr_name="random_walk_pe")(graph)


def _validate_graph(graph, expected_source_idx: int | None = None) -> None:
    if expected_source_idx is not None:
        observed = int(graph.row_id.view(-1)[0])
        if observed != int(expected_source_idx):
            raise RuntimeError(f"row identity changed: {observed} != {expected_source_idx}")
    node_count = int(graph.x.shape[0])
    fragment_count = int(graph.fragment_features.shape[0])
    if graph.fragment_id.shape != (node_count,):
        raise RuntimeError("fragment_id is not atom-aligned")
    if graph.fragment_count.view(-1).tolist() != [fragment_count]:
        raise RuntimeError("fragment_count does not match fragment features")
    if graph.fragment_id.numel() and (
        int(graph.fragment_id.min()) < 0
        or int(graph.fragment_id.max()) >= fragment_count
    ):
        raise RuntimeError("fragment_id is outside the fragment table")
    if graph.fragment_edge_index.shape[0] != 2:
        raise RuntimeError("fragment_edge_index must have shape [2, E]")
    if graph.fragment_edge_attr.shape != (
        graph.fragment_edge_index.shape[1],
        FRAGMENT_EDGE_DIM,
    ):
        raise RuntimeError("fragment edge features are not aligned")
    for value in (
        graph.x,
        graph.edge_attr,
        graph.y,
        graph.random_walk_pe,
        graph.fragment_features,
        graph.fragment_edge_attr,
    ):
        if not torch.isfinite(value.float()).all():
            raise RuntimeError("cache contains non-finite values")


def _source_contract(source_root: Path):
    from rdkit import rdBase

    from .qm9_data import load_qm9_records, prepare_qm9_files

    records = load_qm9_records(source_root)
    files = prepare_qm9_files(source_root)
    return records, files, {
        "source_rows": len(records),
        "rdkit_version": rdBase.rdkitVersion,
        "processed_sha256": sha256_file(files["processed"]),
        "raw_sdf_sha256": sha256_file(files["raw_sdf"]),
    }


def build_cache(
    output_root: Path,
    *,
    source_root: Path,
    source_commit: str,
    shard_size: int = 2_000,
):
    """Build a resumable train/validation-only fragment cache."""
    from rdkit import Chem, RDLogger

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("incomplete final cache manifest exists")

    records, files, source = _source_contract(source_root)
    if source["source_rows"] != 130_831:
        raise RuntimeError(f"unexpected QM9 source rows: {source['source_rows']}")
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(str(files["raw_sdf"]), removeHs=False, sanitize=False)
    validity_path = output_root / "canonical_validity.json"
    if validity_path.is_file():
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        if (
            validity.get("source") != source
            or validity.get("fragment_algorithm") != FRAGMENT_ALGORITHM
        ):
            raise RuntimeError("existing canonical validity belongs to another source")
        valid_pool = np.asarray(validity["valid_source_indices"], dtype=np.int64)
        invalid = validity.get("invalid_records", [])
        train_indices = np.asarray(validity["selected_train_indices"], dtype=np.int64)
        validation_indices = np.asarray(
            validity["selected_validation_indices"], dtype=np.int64
        )
    else:
        valid_pool, invalid = _canonical_valid_pool(records, supplier)
        train_indices, validation_indices = _fixed_split_from_pool(valid_pool)
        validity = {
            "format": "molgap-qm9-fragment-state-canonical-validity-v1",
            "source": source,
            "source_commit": source_commit,
            "fragment_algorithm": FRAGMENT_ALGORITHM,
            "valid_source_indices": valid_pool.tolist(),
            "invalid_records": invalid,
            "selected_train_indices": train_indices.tolist(),
            "selected_validation_indices": validation_indices.tolist(),
            "valid_source_indices_sha256": _indices_sha256(valid_pool),
            "selected_train_indices_sha256": _indices_sha256(train_indices),
            "selected_validation_indices_sha256": _indices_sha256(validation_indices),
            "test_role_materialized": False,
            "official_pcqm_roles_read": False,
        }
        _atomic_json(validity_path, validity)

    roles = {"train": train_indices, "validation": validation_indices}
    progress_path = output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"completed_shards": [], "failures": []}
    )
    completed = {
        (item["role"], item["file"]): item for item in progress["completed_shards"]
    }
    shards = []
    for role, indices in roles.items():
        for part_number, start in enumerate(range(0, len(indices), shard_size)):
            stop = min(start + shard_size, len(indices))
            filename = f"{role}_part_{part_number:03d}.pt"
            path = output_root / filename
            expected = np.asarray(indices[start:stop], dtype=np.int64)
            if path.is_file():
                graphs = _torch_load(path)
            else:
                graphs = [
                    _make_graph(records[int(source_idx)], int(source_idx), supplier)
                    for source_idx in expected
                ]
                _atomic_torch_save(path, graphs)
            if len(graphs) != len(expected):
                raise RuntimeError(f"shard count changed: {filename}")
            for graph, source_idx in zip(graphs, expected):
                _validate_graph(graph, int(source_idx))
            shard = {
                "role": role,
                "file": filename,
                "graph_count": len(graphs),
                "sha256": sha256_file(path),
            }
            old = completed.get((role, filename))
            if old and old.get("sha256") != shard["sha256"]:
                raise RuntimeError(f"resumed shard hash changed: {filename}")
            shards.append(shard)
            _atomic_json(
                progress_path,
                {
                    "format": f"{CACHE_FORMAT}-progress",
                    "source": source,
                    "source_commit": source_commit,
                    "fragment_algorithm": FRAGMENT_ALGORITHM,
                    "completed_shards": shards,
                    "failures": progress.get("failures", []),
                    "test_role_materialized": False,
                    "official_pcqm_roles_read": False,
                },
            )
            print(
                f"{role} shard {part_number + 1}: {len(graphs)} graphs",
                flush=True,
            )

    aggregate = hashlib.sha256()
    for shard in shards:
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    manifest = {
        "format": CACHE_FORMAT,
        "complete": True,
        "source": source,
        "source_commit": source_commit,
        "fragment_algorithm": FRAGMENT_ALGORITHM,
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
            np.concatenate((train_indices, validation_indices)).astype(np.int64).tobytes()
        ).hexdigest()[:16],
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": RWSE_DIM,
        "fragment_feature_channels": FRAGMENT_FEATURE_DIM,
        "fragment_edge_channels": FRAGMENT_EDGE_DIM,
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "gpu_used": False,
        "model_inference_executed": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def accept_cache(cache_root: Path, *, expected_source_commit: str | None = None):
    manifest_path = cache_root / "manifest.json"
    validity_path = cache_root / "canonical_validity.json"
    if not manifest_path.is_file() or not validity_path.is_file():
        raise FileNotFoundError("cache manifest or canonical validity is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    required = {
        "format": CACHE_FORMAT,
        "complete": True,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "held_out_indices_materialized": False,
        "test_role_materialized": False,
        "official_pcqm_roles_read": False,
        "fragment_algorithm": FRAGMENT_ALGORITHM,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"cache contract changed for {key}")
    if expected_source_commit is not None and manifest.get("source_commit") != expected_source_commit:
        raise RuntimeError("cache source commit does not match submitted code")
    if sha256_file(validity_path) != manifest.get("canonical_validity_sha256"):
        raise RuntimeError("canonical validity hash changed")
    if validity.get("source") != manifest.get("source"):
        raise RuntimeError("source contract changed")
    train = np.asarray(validity.get("selected_train_indices", []), dtype=np.int64)
    validation = np.asarray(validity.get("selected_validation_indices", []), dtype=np.int64)
    pool = np.asarray(validity.get("valid_source_indices", []), dtype=np.int64)
    if (
        len(train) != TRAIN_ROWS
        or len(validation) != VALIDATION_ROWS
        or len(np.unique(pool)) != len(pool)
        or not np.isin(train, pool).all()
        or not np.isin(validation, pool).all()
        or np.intersect1d(train, validation).size
        or _indices_sha256(train) != manifest.get("train_source_indices_sha256")
        or _indices_sha256(validation) != manifest.get("validation_source_indices_sha256")
        or _indices_sha256(pool) != manifest.get("canonical_valid_pool_sha256")
        or validity.get("test_role_materialized") is not False
        or validity.get("official_pcqm_roles_read") is not False
    ):
        raise RuntimeError("canonical validity split contract changed")
    observed = []
    aggregate = hashlib.sha256()
    for shard in manifest.get("shards", []):
        path = cache_root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"cache shard hash changed: {path.name}")
        graphs = _torch_load(path)
        if len(graphs) != shard["graph_count"]:
            raise RuntimeError(f"cache shard count changed: {path.name}")
        for graph in graphs:
            _validate_graph(graph)
            observed.append(int(graph.row_id.view(-1)[0]))
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest.get("aggregate_sha256"):
        raise RuntimeError("cache aggregate hash changed")
    expected = np.concatenate((train, validation)).tolist()
    if observed != expected:
        raise RuntimeError("shard order/identity does not match canonical split")
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "accepted": True,
        "source_commit": manifest.get("source_commit"),
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "canonical_validity_sha256": manifest["canonical_validity_sha256"],
        "graph_count": len(observed),
        "roles": manifest["roles"],
        "gpu_used": False,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_materialized": False,
    }
    _atomic_json(cache_root / "acceptance.json", acceptance)
    return acceptance


def load_cache(cache_root: Path, expected_sha256: str | None = None):
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((cache_root / "acceptance.json").read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT or manifest.get("complete") is not True:
        raise RuntimeError("cache is not complete")
    if acceptance.get("format") != ACCEPTANCE_FORMAT or acceptance.get("accepted") is not True:
        raise RuntimeError("cache has no accepted CPU evidence")
    if expected_sha256 and manifest.get("aggregate_sha256") != expected_sha256:
        raise RuntimeError("cache aggregate SHA changed")
    roles: dict[str, list] = {"train": [], "validation": []}
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = cache_root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"cache shard hash changed: {path.name}")
        payload = _torch_load(path)
        if len(payload) != shard["graph_count"]:
            raise RuntimeError(f"cache shard count changed: {path.name}")
        roles[shard["role"]].extend(payload)
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("cache aggregate SHA changed")
    if {key: len(value) for key, value in roles.items()} != manifest["roles"]:
        raise RuntimeError("cache role counts changed")
    return roles, manifest


class OGBFragmentStateStructuralGPSWrapper(OGBEdgeStateStructuralGPSWrapper):
    """OGB EdgeState GPS9 with sparse BRICS fragment-state exchange."""

    def __init__(
        self,
        *args,
        fragment_channels: int = FRAGMENT_CHANNELS,
        fragment_blocks=FRAGMENT_BLOCKS,
        **kwargs,
    ):
        if int(fragment_channels) != FRAGMENT_CHANNELS:
            raise ValueError("fragment_channels is frozen at 32")
        if tuple(fragment_blocks) != FRAGMENT_BLOCKS:
            raise ValueError("fragment_blocks are frozen at (3, 6, 9)")
        super().__init__(*args, **kwargs)
        hidden_channels = self.head[0].in_features
        self.fragment_blocks = tuple(int(value) for value in fragment_blocks)
        self.fragment_features = nn.Sequential(
            nn.LayerNorm(FRAGMENT_FEATURE_DIM),
            nn.Linear(FRAGMENT_FEATURE_DIM, FRAGMENT_CHANNELS),
            nn.SiLU(),
            nn.Linear(FRAGMENT_CHANNELS, FRAGMENT_CHANNELS),
        )
        self.atom_to_fragment = nn.Linear(hidden_channels, FRAGMENT_CHANNELS, bias=False)
        self.fragment_edge = nn.Linear(FRAGMENT_EDGE_DIM, FRAGMENT_CHANNELS, bias=False)
        self.fragment_update_norm = nn.LayerNorm(2 * FRAGMENT_CHANNELS)
        self.fragment_update_gate = nn.Linear(2 * FRAGMENT_CHANNELS, FRAGMENT_CHANNELS)
        self.fragment_update_value = nn.Sequential(
            nn.Linear(2 * FRAGMENT_CHANNELS, 2 * FRAGMENT_CHANNELS),
            nn.SiLU(),
            nn.Linear(2 * FRAGMENT_CHANNELS, FRAGMENT_CHANNELS),
        )
        self.fragment_output_norm = nn.LayerNorm(FRAGMENT_CHANNELS)
        self.fragment_to_atom = _LowRankGatedProjection(
            FRAGMENT_CHANNELS,
            hidden_channels,
            rank=FRAGMENT_EXCHANGE_RANK,
        )

    @staticmethod
    def _mean_by_fragment(values, fragment_id, fragment_count):
        output = values.new_zeros((fragment_count, values.shape[1]))
        counts = values.new_zeros((fragment_count, 1))
        output.index_add_(0, fragment_id, values)
        counts.index_add_(
            0, fragment_id, values.new_ones((fragment_id.shape[0], 1))
        )
        return output / counts.clamp_min_(1.0)

    @staticmethod
    def _fragment_graph_ids(counts):
        return torch.repeat_interleave(
            torch.arange(counts.shape[0], device=counts.device), counts
        )

    def _validate_fragment_inputs(
        self,
        batch,
        fragment_id,
        fragment_count,
        fragment_features,
        fragment_edge_index,
        fragment_edge_attr,
    ):
        counts = fragment_count.view(-1).long()
        features = fragment_features.float()
        if counts.numel() == 0 or int(counts.min()) <= 0:
            raise ValueError("every molecule must have at least one fragment")
        if int(counts.sum()) != int(features.shape[0]):
            raise ValueError("fragment counts do not match fragment features")
        if features.ndim != 2 or features.shape[1] != FRAGMENT_FEATURE_DIM:
            raise ValueError("fragment feature shape changed")
        if not torch.isfinite(features).all():
            raise ValueError("fragment features contain non-finite values")
        if fragment_id.ndim != 1 or fragment_id.shape[0] != batch.shape[0]:
            raise ValueError("fragment IDs are not atom-aligned")
        if fragment_id.numel() and (
            int(fragment_id.min()) < 0
            or int(fragment_id.max()) >= int(features.shape[0])
        ):
            raise ValueError("fragment ID is out of range")
        graph_ids = self._fragment_graph_ids(counts)
        if fragment_edge_index.ndim != 2 or fragment_edge_index.shape[0] != 2:
            raise ValueError("fragment edge index must have shape [2, E]")
        if fragment_edge_attr.shape != (
            fragment_edge_index.shape[1],
            FRAGMENT_EDGE_DIM,
        ):
            raise ValueError("fragment edge features are not aligned")
        if fragment_edge_index.numel():
            if int(fragment_edge_index.min()) < 0 or int(fragment_edge_index.max()) >= features.shape[0]:
                raise ValueError("fragment edge index is out of range")
            if not torch.equal(
                graph_ids[fragment_edge_index[0]], graph_ids[fragment_edge_index[1]]
            ):
                raise ValueError("fragment edge crosses molecules")
        return counts, features

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        fragment_id,
        fragment_count,
        fragment_features,
        fragment_edge_index,
        fragment_edge_attr,
    ):
        return self.head(
            self.encode(
                x,
                edge_index,
                edge_attr,
                batch,
                random_walk_pe,
                fragment_id,
                fragment_count,
                fragment_features,
                fragment_edge_index,
                fragment_edge_attr,
            )
        )

    def encode(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe,
        fragment_id,
        fragment_count,
        fragment_features,
        fragment_edge_index,
        fragment_edge_attr,
    ):
        if random_walk_pe is None:
            raise ValueError("fragment-state GPS requires random_walk_pe")
        expected = (x.shape[0], self.rwse_dim)
        if tuple(random_walk_pe.shape) != expected or not torch.isfinite(random_walk_pe).all():
            raise ValueError(f"random_walk_pe must be finite with shape {expected}")
        counts, features = self._validate_fragment_inputs(
            batch,
            fragment_id,
            fragment_count,
            fragment_features,
            fragment_edge_index,
            fragment_edge_attr,
        )
        h = self._embed_nodes(x)
        h = h + self.rwse_encoder(random_walk_pe.float())
        edge_state = self._embed_edges(edge_attr)
        fragment_state = self.fragment_features(features)
        for layer, (edge_update, conv) in enumerate(
            zip(self.edge_updates, self.convs), start=1
        ):
            edge_state = edge_update(h, edge_index, edge_state)
            h = conv(h, edge_index, batch, edge_attr=edge_state)
            if layer not in self.fragment_blocks:
                continue
            atom_summary = self._mean_by_fragment(
                self.atom_to_fragment(h), fragment_id, fragment_state.shape[0]
            )
            neighbor = fragment_state.new_zeros(fragment_state.shape)
            degree = fragment_state.new_zeros((fragment_state.shape[0], 1))
            if fragment_edge_index.shape[1]:
                source, target = fragment_edge_index
                messages = fragment_state[source] + self.fragment_edge(
                    fragment_edge_attr.float()
                )
                neighbor.index_add_(0, target, messages)
                degree.index_add_(
                    0, target, fragment_state.new_ones((target.shape[0], 1))
                )
            neighbor = neighbor / degree.clamp_min_(1.0)
            combined = self.fragment_update_norm(
                torch.cat([fragment_state, atom_summary + neighbor], dim=-1)
            )
            gate = torch.sigmoid(self.fragment_update_gate(combined))
            proposal = self.fragment_update_value(combined)
            fragment_state = self.fragment_output_norm(
                fragment_state + gate * proposal
            )
            h = h + self.fragment_to_atom(fragment_state[fragment_id])
        return self._pool(h, batch)


def make_model(kind: str):
    kwargs = {
        "in_channels": 9,
        "edge_dim": 3,
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "n_targets": 1,
        "pooling": "mean",
        "rwse_dim": RWSE_DIM,
        "edge_state_channels": 64,
    }
    if kind == "control":
        return OGBEdgeStateStructuralGPSWrapper(**kwargs)
    if kind == "fragment_state":
        return OGBFragmentStateStructuralGPSWrapper(**kwargs)
    raise ValueError(f"unknown model kind: {kind}")


def forward_model(model, batch, kind: str):
    if kind == "control":
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        ).view(-1)
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.fragment_id,
        batch.fragment_count,
        batch.fragment_features,
        batch.fragment_edge_index,
        batch.fragment_edge_attr,
    ).view(-1)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_sha256(model, *, common_only: bool = False, other=None) -> str:
    digest = hashlib.sha256()
    values = model.state_dict()
    other_values = other.state_dict() if other is not None else None
    for name, value in sorted(values.items()):
        if common_only and (other_values is None or name not in other_values):
            continue
        if other_values is not None and common_only and not torch.equal(value, other_values[name]):
            raise RuntimeError(f"matched initialization differs at {name}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _rng_state(loader):
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": loader.generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state, loader):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    loader.generator.set_state(state["loader"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _loader(graphs, *, shuffle: bool, seed: int):
    from torch_geometric.loader import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _target_stats(graphs):
    values = torch.stack([graph.y.view(()) for graph in graphs])
    return values.mean(), values.std().clamp_min(1e-6)


def _evaluate(model, kind, loader, mean, std, device):
    model.eval()
    absolute = 0.0
    rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = forward_model(model, batch, kind) * std + mean
            absolute += float((prediction - batch.y.view(-1)).abs().sum())
            rows += int(prediction.numel())
    return absolute / rows


def _train_one(
    kind: str,
    roles: dict[str, list],
    output_dir: Path,
    *,
    source_commit: str,
    cache_sha256: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("SCNet did not expose a CUDA/DCU device")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    mean, std = _target_stats(roles["train"])
    mean, std = mean.to(device), std.to(device)
    set_seed(MODEL_SEED)
    model = make_model(kind).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    train_loader = _loader(roles["train"], shuffle=True, seed=MODEL_SEED)
    validation_loader = _loader(roles["validation"], shuffle=False, seed=MODEL_SEED)
    last_path = output_dir / "last_checkpoint.pt"
    trace = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    start_epoch = 0
    if last_path.is_file():
        checkpoint = _torch_load(last_path, map_location=device)
        expected = {
            "format": SCREEN_FORMAT,
            "kind": kind,
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": MODEL_SEED,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{kind} checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        best = float(checkpoint["best"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], train_loader)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model, batch, kind)
            target = (batch.y.view(-1) - mean) / std
            loss = torch.nn.functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float((prediction.detach() * std + mean - batch.y.view(-1)).abs().sum())
            rows += int(prediction.numel())
        validation = _evaluate(model, kind, validation_loader, mean, std, device)
        improved = validation < best
        if improved:
            best = validation
            best_epoch = epoch
            stale = 0
            _atomic_torch_save(output_dir / "best_model.pt", _cpu_state(model))
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": absolute / rows,
            "validation_gap_mae_eV": validation,
            "seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
        }
        trace.append(row)
        scheduler.step()
        _atomic_torch_save(
            last_path,
            {
                "format": SCREEN_FORMAT,
                "kind": kind,
                "epoch": epoch,
                "model": _cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": MODEL_SEED,
                "best": best,
                "best_epoch": best_epoch,
                "stale": stale,
                "rng": _rng_state(train_loader),
            },
        )
        _atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"{kind} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"val={validation:.6f}eV {row['seconds']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    if not (output_dir / "best_model.pt").is_file():
        raise RuntimeError(f"{kind} produced no best checkpoint")
    return {
        "kind": kind,
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "model_sha256": sha256_file(output_dir / "best_model.pt"),
        "checkpoint_sha256": sha256_file(last_path),
    }


def run_preflight(cache_root: Path, output_root: Path, *, source_commit: str, cache_sha256: str):
    roles, manifest = load_cache(cache_root, cache_sha256)
    if len(roles["train"]) < BATCH_SIZE:
        raise RuntimeError("preflight needs a physical batch of 128 graphs")
    if not torch.cuda.is_available():
        raise RuntimeError("SCNet did not expose a CUDA/DCU device")
    output_root.mkdir(parents=True, exist_ok=True)
    batch = next(iter(_loader(roles["train"][:BATCH_SIZE], shuffle=False, seed=MODEL_SEED))).to("cuda")
    reports = []
    for kind in ("control", "fragment_state"):
        torch.cuda.reset_peak_memory_stats()
        set_seed(MODEL_SEED)
        model = make_model(kind).to("cuda")
        model.train()
        prediction = forward_model(model, batch, kind)
        loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1))
        loss.backward()
        parameters = list(model.parameters())
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in parameters
        )
        reports.append(
            {
                "kind": kind,
                "accepted": finite,
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "batch_size": BATCH_SIZE,
                "node_count": int(batch.x.shape[0]),
                "fragment_count": int(batch.fragment_features.shape[0]),
                "finite_forward_backward": finite,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            }
        )
        del model
        torch.cuda.empty_cache()
    result = {
        "format": f"{SCREEN_FORMAT}-preflight",
        "accepted": all(item["accepted"] for item in reports),
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "cuda_available": True,
        "batch_size": BATCH_SIZE,
        "reports": reports,
        "official_pcqm_roles_read": False,
        "test_role_materialized": False,
    }
    _atomic_json(output_root / "preflight.json", result)
    if not result["accepted"]:
        raise RuntimeError("fragment-state preflight produced non-finite gradients")
    return result


def run_screen(
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
    preflight_path: Path,
):
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("accepted") is not True
        or preflight.get("format") != f"{SCREEN_FORMAT}-preflight"
        or preflight.get("batch_size") != BATCH_SIZE
        or preflight.get("source_commit") != source_commit
        or preflight.get("cache_aggregate_sha256") != cache_sha256
        or preflight.get("official_pcqm_roles_read") is not False
        or preflight.get("test_role_materialized") is not False
    ):
        raise RuntimeError("preflight contract is not accepted")
    roles, manifest = load_cache(cache_root, cache_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(MODEL_SEED)
    control = make_model("control")
    control_initial_sha = _state_sha256(control)
    set_seed(MODEL_SEED)
    candidate = make_model("fragment_state")
    common_initial_sha = _state_sha256(candidate, common_only=True, other=control)
    if not common_initial_sha:
        raise RuntimeError("control/candidate have no shared initialized parameters")
    del control, candidate
    results = {
        "control": _train_one(
            "control",
            roles,
            output_root / "control",
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        ),
        "fragment_state": _train_one(
            "fragment_state",
            roles,
            output_root / "fragment_state",
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        ),
    }
    control_mae = results["control"]["validation_gap_mae_eV"]
    candidate_mae = results["fragment_state"]["validation_gap_mae_eV"]
    gain = control_mae - candidate_mae
    nominated = bool(gain >= MIN_GAIN_EV and candidate_mae < control_mae)
    summary = {
        "format": SCREEN_FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "split_fingerprint": manifest["split_fingerprint"],
        "seed": MODEL_SEED,
        "gpu": torch.cuda.get_device_name(0),
        "architecture": {
            "control": "ogb_edge_state_structural_gps9",
            "candidate": "ogb_edge_state_structural_gps9_brics_fragment_state",
            "hidden_channels": 192,
            "num_layers": 9,
            "num_heads": 4,
            "rwse_dim": RWSE_DIM,
            "edge_state_channels": 64,
            "fragment_channels": FRAGMENT_CHANNELS,
            "fragment_blocks": list(FRAGMENT_BLOCKS),
            "fragment_algorithm": FRAGMENT_ALGORITHM,
        },
        "control_initial_sha256": control_initial_sha,
        "shared_initial_sha256": common_initial_sha,
        "training_contract": {
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "precision": "fp32",
            "target": "gap_eV",
        },
        "results": results,
        "paired_gain_eV": gain,
        "minimum_gain_eV": MIN_GAIN_EV,
        "pcqm_transfer_nominated": nominated,
        "official_pcqm_roles_read": False,
        "test_role_materialized": False,
    }
    _atomic_json(output_root / "metrics.json", summary)
    completion = {
        **summary,
        "artifact_sha256": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and path.name != "completion_manifest.json"
        },
    }
    _atomic_json(output_root / "completion_manifest.json", completion)
    return summary


__all__ = [
    "ACCEPTANCE_FORMAT",
    "BATCH_SIZE",
    "CACHE_FORMAT",
    "EPOCHS",
    "FRAGMENT_BLOCKS",
    "FRAGMENT_CHANNELS",
    "FRAGMENT_FEATURE_DIM",
    "FRAGMENT_ALGORITHM",
    "MIN_GAIN_EV",
    "SCREEN_FORMAT",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "OGBFragmentStateStructuralGPSWrapper",
    "accept_cache",
    "build_cache",
    "forward_model",
    "load_cache",
    "make_fragment_data_class",
    "make_model",
    "run_preflight",
    "run_screen",
]
