"""Reproducible QM9 architecture-screen data and training utilities."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .egnn import EGNNWrapper
from .edge_global_2d import EdgeGlobal2DWrapper
from .gine import GINEWrapper
from .gps import (
    EdgeConditionedStructuralGPSWrapper,
    DirectedEdgeStateStructuralGPSWrapper,
    EdgeJKReadoutStructuralGPSWrapper,
    EdgeReadoutStructuralGPSWrapper,
    EdgeStateStructuralGPSWrapper,
    FrontierCenterGapHead,
    GraphTokenStructuralGPSWrapper,
    GPSWrapper,
    SparsePathAttentionStructuralGPSWrapper,
)
from .graphs import smiles_to_pyg
from .geometry_features import (
    ANGLE_FEATURE_DIM,
    FULL_FEATURE_DIM,
    local_geometry_features,
    select_geometry_features,
)
from .schnet import SchNetWrapper
from .tensornet import TensorNetWrapper
from .tgt_lite import TGTLiteWrapper
from .tgt_hybrid import TGTLiteHybridWrapper
from .tgt_hybrid_v2 import TGTLiteHybridV2Wrapper
from .pair_triplet_2d import PairTriplet2DWrapper
from .pair_triplet_2d_rich import PairTriplet2DRichWrapper
from .pair_gps_2d import (
    PairGPS2DR2Wrapper,
    PairGPS2DR3Wrapper,
    PairGPS2DWrapper,
)
from .structural_encoding import build_rwse_graph_cache, sha256
from .tgt_egt_hybrid import TGTEGTHybridWrapper
from .tgt_egt_compact import TGTCompactEGTWrapper
from .tgt_egt_rich import TGTEGTRichWrapper
from .tgt_egt_hybrid_plus import TGTEGTHybridPlusWrapper
from .tgt_egt_hybrid_warmblend import TGTEGTHybridWarmBlendWrapper

QM9_PROCESSED_URL = "https://data.pyg.org/datasets/qm9_v3.zip"
QM9_RAW_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/molnet_publish/qm9.zip"
)
TARGET_NAMES = ("HOMO", "LUMO", "Gap")
TARGET_COLUMNS = (2, 3, 4)
DEFAULT_CACHE = Path("data/cache/qm9")
DEFAULT_RESULTS = Path("experiments/qm9_architecture/results")
DEFAULT_MODELS = Path("models/experiments/qm9_architecture_screen")

ENCODER_CONFIGS = {
    "gine6": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 6,
        "dropout": 0.05,
        "batch_size": 256,
    },
    "gps7": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 7,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 256,
    },
    "gps9": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_160": {
        "kind": "topology",
        "hidden_channels": 160,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_128": {
        "kind": "topology",
        "hidden_channels": 128,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_meanmax": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean_max",
        "batch_size": 192,
    },
    "gps11_160": {
        "kind": "topology",
        "hidden_channels": 160,
        "num_layers": 11,
        "num_heads": 4,
        "dropout": 0.05,
        "batch_size": 192,
    },
    "gps9_pcqm_transfer": {
        "kind": "topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "input_channels": 18,
        "batch_size": 192,
    },
    "edge_global_2d": {
        "kind": "topology",
        "hidden_channels": 192,
        "edge_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean_max",
        "batch_size": 128,
    },
    "pair_triplet_2d": {
        "kind": "topology",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 6,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean_max",
        "path_steps": 3,
        "triplet_rank": 8,
        "batch_size": 96,
    },
    "pair_triplet_2d_rich": {
        "kind": "topology",
        "hidden_channels": 256,
        "pair_channels": 96,
        "num_layers": 10,
        "num_heads": 8,
        "dropout": 0.05,
        "pooling": "mean_max",
        "path_steps": 5,
        "triplet_rank": 16,
        "batch_size": 48,
    },
    "pair_gps_2d": {
        "kind": "topology",
        "hidden_channels": 256,
        "pair_channels": 96,
        "num_layers": 10,
        "num_heads": 8,
        "dropout": 0.05,
        "pooling": "mean",
        "path_steps": 5,
        "triplet_rank": 16,
        "batch_size": 48,
        "amp": False,
    },
    "pair_gps_2d_r2": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "distance_cap": 5,
        "triplet_rank": 8,
        "triplet_interval": 3,
        "rwse_dim": 16,
        "gate_init": 0.1,
        "batch_size": 48,
        "amp": False,
    },
    "edge_state_structural_gps": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "batch_size": 48,
        "amp": False,
    },
    "edge_state_structural_orbital": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "consistent_head": True,
        "batch_size": 48,
        "amp": False,
    },
    "edge_state_structural_readout": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "readout_channels": 32,
        "batch_size": 48,
        "amp": False,
    },
    "edge_state_structural_jk_readout": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "readout_layers": (3, 6, 9),
        "readout_channels": 32,
        "batch_size": 48,
        "amp": False,
    },
    "edge_conditioned_structural_gps": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "batch_size": 48,
        "amp": False,
    },
    "graph_token_structural_gps": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "token_channels": 16,
        "batch_size": 48,
        "amp": False,
    },
    "multihop_edge_state_structural_gps": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "model_edge_dim": 8,
        "multihop_max_distance": 4,
        "batch_size": 48,
        "amp": False,
    },
    "sparse_path_attention_structural_gps": {
        "kind": "structural_multihop",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "path_attention_rank": 16,
        "path_max_distance": 4,
        "multihop_attention_distance": 4,
        "batch_size": 48,
        "amp": False,
    },
    "directed_edge_state_structural_gps": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "rwse_dim": 16,
        "edge_state_channels": 64,
        "batch_size": 48,
        "amp": False,
    },
    "pair_gps_2d_r3_orbital": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "distance_cap": 5,
        "triplet_rank": 8,
        "triplet_interval": 3,
        "rwse_dim": 16,
        "gate_init": 0.1,
        "attentive_triplet": False,
        "consistent_head": True,
        "batch_size": 48,
        "amp": False,
    },
    "pair_gps_2d_r3_triplet": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "distance_cap": 5,
        "triplet_rank": 8,
        "triplet_interval": 3,
        "rwse_dim": 16,
        "gate_init": 0.1,
        "attentive_triplet": True,
        "consistent_head": False,
        "batch_size": 48,
        "amp": False,
    },
    "pair_gps_2d_r3_combined": {
        "kind": "structural_topology",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 9,
        "num_heads": 4,
        "dropout": 0.05,
        "pooling": "mean",
        "distance_cap": 5,
        "triplet_rank": 8,
        "triplet_interval": 3,
        "rwse_dim": 16,
        "gate_init": 0.1,
        "attentive_triplet": True,
        "consistent_head": True,
        "batch_size": 48,
        "amp": False,
    },
    "tgt_egt_hybrid": {
        "kind": "hybrid_egt",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 9,
        "dropout": 0.05,
        "batch_size": 32,
    },
    "tgt_egt_compact": {
        "kind": "geometry",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 32,
        "cutoff": 12.0,
        "dropout": 0.05,
        "batch_size": 96,
    },
    "tgt_egt_stable": {
        "kind": "geometry",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 32,
        "cutoff": 12.0,
        "dropout": 0.05,
        "zero_init_bond_channels": True,
        "batch_size": 96,
    },
    "tgt_egt_rich": {
        "kind": "hybrid_egt_rich",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 8,
        "topology_hidden_channels": 256,
        "topology_pair_channels": 96,
        "topology_heads": 8,
        "topology_path_steps": 5,
        "topology_triplet_rank": 16,
        "dropout": 0.05,
        "learning_rate": 2e-4,
        "amp": False,
        "batch_size": 16,
    },
    "tgt_egt_hybrid_plus": {
        "kind": "hybrid_egt_plus",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 9,
        "expert_hidden_channels": 192,
        "expert_pair_channels": 64,
        "expert_layers": 6,
        "expert_heads": 4,
        "expert_path_steps": 5,
        "expert_triplet_rank": 8,
        "dropout": 0.05,
        "learning_rate": 2e-4,
        "amp": False,
        "batch_size": 16,
    },
    "tgt_egt_hybrid_frozen": {
        "kind": "hybrid_egt_plus",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 9,
        "expert_hidden_channels": 192,
        "expert_pair_channels": 64,
        "expert_layers": 6,
        "expert_heads": 4,
        "expert_path_steps": 5,
        "expert_triplet_rank": 8,
        "dropout": 0.05,
        "learning_rate": 5e-4,
        "amp": False,
        "freeze_base": True,
        "batch_size": 16,
    },
    "tgt_egt_hybrid_warmblend": {
        "kind": "hybrid_egt_warmblend",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 9,
        "expert_hidden_channels": 256,
        "expert_pair_channels": 96,
        "expert_layers": 10,
        "expert_heads": 8,
        "expert_path_steps": 5,
        "expert_triplet_rank": 16,
        "dropout": 0.05,
        "learning_rate": 1e-4,
        "amp": False,
        "batch_size": 16,
    },
    "tgt_egt_hybrid_warmblend_frozen": {
        "kind": "hybrid_egt_warmblend",
        "hidden_channels": 192,
        "pair_channels": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 12.0,
        "topology_layers": 9,
        "expert_hidden_channels": 256,
        "expert_pair_channels": 96,
        "expert_layers": 10,
        "expert_heads": 8,
        "expert_path_steps": 5,
        "expert_triplet_rank": 16,
        "dropout": 0.05,
        "learning_rate": 5e-4,
        "amp": False,
        "freeze_base": True,
        "freeze_expert": True,
        "batch_size": 16,
    },
    "schnet": {
        "kind": "geometry",
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
        "batch_size": 128,
    },
    "schnet_angle": {
        "kind": "geometry",
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
        "n_atom_geom": ANGLE_FEATURE_DIM,
        "atom_geom_mode": "angle",
        "batch_size": 128,
    },
    "schnet_angle_dihedral": {
        "kind": "geometry",
        "hidden_channels": 176,
        "num_filters": 160,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "dropout": 0.05,
        "n_atom_geom": FULL_FEATURE_DIM,
        "atom_geom_mode": "angle_dihedral",
        "batch_size": 128,
    },
    "tensornet": {
        "kind": "geometry",
        "hidden_channels": 128,
        "num_layers": 2,
        "num_rbf": 32,
        "cutoff": 5.0,
        "dropout": 0.0,
        "batch_size": 32,
    },
    "egnn": {
        "kind": "geometry",
        "hidden_channels": 128,
        "num_layers": 4,
        "num_rbf": 32,
        "cutoff": 5.0,
        "dropout": 0.05,
        "batch_size": 128,
    },
    "tgt_lite": {
        "kind": "geometry",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 32,
        "cutoff": 12.0,
        "dropout": 0.05,
        "batch_size": 96,
    },
    "tgt_hybrid": {
        "kind": "hybrid",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 32,
        "cutoff": 12.0,
        "topology_layers": 9,
        "dropout": 0.05,
        "batch_size": 64,
    },
    "tgt_hybrid_v2": {
        "kind": "hybrid_v2",
        "hidden_channels": 192,
        "pair_channels": 48,
        "num_layers": 8,
        "num_heads": 4,
        "num_rbf": 32,
        "cutoff": 12.0,
        "topology_layers": 6,
        "dropout": 0.05,
        "batch_size": 48,
    },
}


@dataclass(frozen=True)
class ScreenSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int

    @property
    def all_indices(self) -> np.ndarray:
        return np.concatenate((self.train, self.validation, self.test))

    @property
    def fingerprint(self) -> str:
        value = self.all_indices.astype(np.int64).tobytes()
        return hashlib.sha256(value).hexdigest()[:16]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_split(
    n_total: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> ScreenSplit:
    requested = train_size + validation_size + test_size
    if requested > n_total:
        raise ValueError(f"Requested {requested} rows from QM9 with {n_total} rows")
    order = np.random.RandomState(seed).permutation(n_total)[:requested]
    train_end = train_size
    validation_end = train_end + validation_size
    return ScreenSplit(
        train=order[:train_end],
        validation=order[train_end:validation_end],
        test=order[validation_end:],
        seed=seed,
    )


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def prepare_qm9_processed(cache_dir: Path = DEFAULT_CACHE) -> Path:
    processed = cache_dir / "preprocessed" / "qm9_v3.pt"
    if not processed.exists():
        archive = cache_dir / "download" / "qm9_v3.zip"
        _download(QM9_PROCESSED_URL, archive)
        processed.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(processed.parent)
    return processed


def prepare_qm9_files(cache_dir: Path = DEFAULT_CACHE) -> dict[str, Path]:
    processed = prepare_qm9_processed(cache_dir)
    raw_sdf = cache_dir / "raw" / "gdb9.sdf"
    if not raw_sdf.exists():
        archive = cache_dir / "download" / "qm9_raw.zip"
        _download(QM9_RAW_URL, archive)
        raw_sdf.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(raw_sdf.parent)
    return {"processed": processed, "raw_sdf": raw_sdf}


def load_qm9_records(cache_dir: Path = DEFAULT_CACHE) -> list[dict]:
    processed = prepare_qm9_processed(cache_dir)
    records = torch.load(processed, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Unexpected QM9 payload: {processed}")
    return records


def target_tensor(record: dict) -> torch.Tensor:
    return record["y"].view(-1)[list(TARGET_COLUMNS)].float()


def target_stats(records: list[dict], train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.stack([target_tensor(records[int(i)]) for i in train_indices])
    mean = targets.mean(dim=0)
    std = targets.std(dim=0).clamp_min(1e-6)
    return mean, std


def frontier_center_stats(
    records: list[dict], train_indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.stack([target_tensor(records[int(i)]) for i in train_indices])
    centers = 0.5 * (targets[:, 0] + targets[:, 1])
    return centers.mean().view(1), centers.std().clamp_min(1e-6).view(1)


def configure_frontier_head(
    model,
    records: list[dict],
    train_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict | None:
    if not isinstance(model.head, FrontierCenterGapHead):
        return None
    center_mean, center_std = frontier_center_stats(records, train_indices)
    model.head.configure_target_stats(mean, std, center_mean, center_std)
    return {
        "constraint": "lumo_minus_homo_equals_gap",
        "center_mean": center_mean.tolist(),
        "center_std": center_std.tolist(),
    }


def _topology_graph(record: dict, source_idx: int, mean: torch.Tensor, std: torch.Tensor) -> Data:
    target = target_tensor(record)
    return Data(
        x=record["x"].float(),
        z=record["z"].long(),
        edge_index=record["edge_index"].long(),
        edge_attr=record["edge_attr"].float(),
        y=((target - mean) / std).view(1, -1),
        y_eV=target.view(1, -1),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )


def _dft_graph(record: dict, source_idx: int, mean: torch.Tensor, std: torch.Tensor) -> Data:
    target = target_tensor(record)
    return Data(
        x=record["x"].float(),
        edge_index=record["edge_index"].long(),
        edge_attr=record["edge_attr"].float(),
        z=record["z"].long(),
        pos=record["pos"].float(),
        y=((target - mean) / std).view(1, -1),
        y_eV=target.view(1, -1),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )


def build_etkdg_cache(
    records: list[dict],
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    cache_dir: Path = DEFAULT_CACHE,
    seed: int = 42,
    mmff_iters: int = 200,
    shard_size: int = 2000,
) -> tuple[dict[int, Data], dict]:
    paths = prepare_qm9_files(cache_dir)
    protocol = f"qm9-etkdg-v3-sanitize-false-mmff{mmff_iters}".encode()
    cache_identity = protocol + b"\0" + indices.astype(np.int64).tobytes()
    key = hashlib.sha256(cache_identity).hexdigest()[:16]
    output = cache_dir / "etkdg" / f"graphs_{key}_seed{seed}.pt"
    report_path = output.with_suffix(".json")
    if output.exists() and report_path.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        return payload, json.loads(report_path.read_text(encoding="utf-8"))

    from rdkit import Chem, RDLogger

    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / "shards" / f"{key}_seed{seed}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(
        str(paths["raw_sdf"]), removeHs=False, sanitize=False
    )
    graphs: dict[int, Data] = {}
    failures: list[int] = []
    started = time.perf_counter()
    resumed_shards = 0
    index_list = indices.tolist()
    for start in range(0, len(index_list), shard_size):
        stop = min(start + shard_size, len(index_list))
        shard_path = shard_dir / f"{start:06d}_{stop:06d}.pt"
        if shard_path.exists():
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_graphs = shard["graphs"]
            shard_failures = shard["failure_indices"]
            resumed_shards += 1
        else:
            shard_graphs: dict[int, Data] = {}
            shard_failures: list[int] = []
            for source_idx in index_list[start:stop]:
                record = records[source_idx]
                raw_idx = int(str(record["name"]).split("_")[-1]) - 1
                mol = supplier[raw_idx]
                if mol is None:
                    shard_failures.append(source_idx)
                    continue
                try:
                    no_hydrogen = Chem.RemoveHs(mol, sanitize=False)
                    smiles = Chem.MolToSmiles(
                        no_hydrogen, canonical=True, isomericSmiles=True
                    )
                    if Chem.MolFromSmiles(smiles) is None:
                        shard_failures.append(source_idx)
                        continue
                except Exception:
                    shard_failures.append(source_idx)
                    continue
                graph = smiles_to_pyg(
                    smiles,
                    use_charges=False,
                    mmff_iters=mmff_iters,
                    random_seed=(seed * 1_000_003 + source_idx) % 2_147_483_647,
                )
                if graph is None:
                    shard_failures.append(source_idx)
                    continue
                target = target_tensor(record)
                graph.y = ((target - mean) / std).view(1, -1)
                graph.y_eV = target.view(1, -1)
                graph.x = record["x"].float()
                graph.edge_index = record["edge_index"].long()
                graph.edge_attr = record["edge_attr"].float()
                graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
                shard_graphs[source_idx] = graph
            _atomic_torch_save(
                shard_path,
                {
                    "graphs": shard_graphs,
                    "failure_indices": shard_failures,
                    "start": start,
                    "stop": stop,
                },
            )
        graphs.update(shard_graphs)
        failures.extend(shard_failures)
        print(
            f"ETKDG {stop}/{len(indices)} success={len(graphs)} "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )

    report = {
        "requested": int(len(indices)),
        "succeeded": len(graphs),
        "failed": len(failures),
        "failure_indices": failures,
        "seed": seed,
        "mmff_iters": mmff_iters,
        "shard_size": shard_size,
        "resumed_shards": resumed_shards,
        "sdf_sanitize": False,
        "cache_version": 3,
        "elapsed_s": time.perf_counter() - started,
        "index_sha256": hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
    }
    _atomic_torch_save(output, graphs)
    _atomic_json(report_path, report)
    return graphs, report


def make_graph_splits(
    records: list[dict],
    split: ScreenSplit,
    geometry: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    cache_dir: Path,
    seed: int,
) -> tuple[dict[str, list[Data]], dict]:
    roles = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    if geometry == "topology":
        return {
            role: [_topology_graph(records[int(i)], int(i), mean, std) for i in indices]
            for role, indices in roles.items()
        }, {"geometry": "topology", "failed": 0}
    if geometry == "dft":
        return {
            role: [_dft_graph(records[int(i)], int(i), mean, std) for i in indices]
            for role, indices in roles.items()
        }, {"geometry": "dft", "failed": 0}
    if geometry != "etkdg":
        raise ValueError(f"Unsupported geometry: {geometry}")
    graphs, report = build_etkdg_cache(
        records, split.all_indices, mean, std, cache_dir=cache_dir, seed=seed
    )
    graph_splits = {
        role: [graphs[int(i)] for i in indices if int(i) in graphs]
        for role, indices in roles.items()
    }
    # Older ETKDG cache shards predate the pair/triplet candidate and contain
    # only z/pos.  Attach the immutable processed 2D view in memory so a cache
    # migration never changes the geometry or split identity.
    for role_graphs in graph_splits.values():
        for graph in role_graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            if not hasattr(graph, "x"):
                graph.x = records[source_idx]["x"].float()
                graph.edge_index = records[source_idx]["edge_index"].long()
                graph.edge_attr = records[source_idx]["edge_attr"].float()
            # ETKDG nodes/coordinates and the legacy 2D view can have
            # different node counts.  Keep a separately collatable 2D view
            # so hybrid encoders never reuse the 3D batch vector for it.
            graph.topology_x = graph.x.float().contiguous()
            graph.topology_edges = graph.edge_index.t().contiguous()
            graph.topology_edge_attr = graph.edge_attr.float().contiguous()
            graph.topology_node_count = torch.tensor(
                [graph.x.shape[0]], dtype=torch.long
            )
            graph.topology_edge_count = torch.tensor(
                [graph.edge_index.shape[1]], dtype=torch.long
            )
            graph.geometry_node_count = torch.tensor(
                [graph.z.shape[0]], dtype=torch.long
            )
    return graph_splits, {"geometry": "etkdg", **report}


def qm9_rwse_cache_paths(
    cache_dir: Path,
    split: ScreenSplit,
    *,
    walk_length: int = 16,
) -> dict[str, Path]:
    root = cache_dir / "structural" / (
        f"topology_{split.fingerprint}_rwse{walk_length}_v1"
    )
    return {
        "input": root / "topology_graphs.pt",
        "output": root / "topology_graphs_rwse.pt",
        "progress": root / "parts",
        "acceptance": root / "acceptance.json",
    }


def build_qm9_rwse_screen_cache(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int = 42,
    walk_length: int = 16,
    shard_size: int = 5_000,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    """Build and accept the CPU-only RWSE cache required by structural screens."""
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    expected_indices = split.all_indices.astype(np.int64)
    # RWSE depends only on topology.  In particular, this CPU stage does not
    # copy validation/test labels into the accepted GPU input cache.
    ordered_graphs = [
        Data(
            x=records[int(source_idx)]["x"].float(),
            edge_index=records[int(source_idx)]["edge_index"].long(),
            edge_attr=records[int(source_idx)]["edge_attr"].float(),
            source_idx=torch.tensor([int(source_idx)], dtype=torch.long),
        )
        for source_idx in expected_indices
    ]
    paths = qm9_rwse_cache_paths(
        cache_dir, split, walk_length=walk_length
    )
    if paths["input"].is_file():
        existing = torch.load(
            paths["input"], map_location="cpu", weights_only=False
        )
        existing_indices = np.asarray(
            [int(graph.source_idx.view(-1)[0]) for graph in existing],
            dtype=np.int64,
        )
        if not np.array_equal(existing_indices, expected_indices):
            raise ValueError("Existing QM9 RWSE input has a different split order")
    else:
        _atomic_torch_save(paths["input"], ordered_graphs)

    manifest = build_rwse_graph_cache(
        paths["input"],
        paths["output"],
        paths["progress"],
        walk_length=walk_length,
        shard_size=shard_size,
    )
    encoded = torch.load(paths["output"], map_location="cpu", weights_only=False)
    encoded_indices = np.asarray(
        [int(graph.source_idx.view(-1)[0]) for graph in encoded],
        dtype=np.int64,
    )
    if not np.array_equal(encoded_indices, expected_indices):
        raise RuntimeError("RWSE output source indices do not match the fixed split")
    for graph in encoded:
        positional = getattr(graph, "random_walk_pe", None)
        if positional is None or positional.shape != (graph.num_nodes, walk_length):
            raise RuntimeError("RWSE output contains a missing or malformed encoding")
        if not torch.isfinite(positional).all():
            raise RuntimeError("RWSE output contains non-finite values")
    acceptance = {
        "format": "molgap-qm9-rwse-acceptance-v1",
        "complete": True,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "index_sha256": hashlib.sha256(expected_indices.tobytes()).hexdigest(),
        "roles": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "walk_length": walk_length,
        "rows": len(encoded),
        "output_path": str(paths["output"]),
        "output_sha256": sha256(paths["output"]),
        "cache_manifest": str(paths["output"].with_suffix(".manifest.json")),
        "parts": len(manifest["parts"]),
    }
    _atomic_json(paths["acceptance"], acceptance)
    return acceptance


def attach_accepted_qm9_rwse(
    graph_splits: dict[str, list[Data]],
    *,
    cache_dir: Path,
    split: ScreenSplit,
    walk_length: int = 16,
) -> dict:
    """Attach an accepted cache, refusing any silent GPU-side construction."""
    paths = qm9_rwse_cache_paths(
        cache_dir, split, walk_length=walk_length
    )
    if not paths["acceptance"].is_file() or not paths["output"].is_file():
        raise FileNotFoundError(
            "Accepted QM9 RWSE cache is required; run the CPU build-rwse step first"
        )
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    expected_indices = split.all_indices.astype(np.int64)
    expected_contract = {
        "format": "molgap-qm9-rwse-acceptance-v1",
        "complete": True,
        "split_seed": split.seed,
        "split_fingerprint": split.fingerprint,
        "index_sha256": hashlib.sha256(expected_indices.tobytes()).hexdigest(),
        "walk_length": walk_length,
        "rows": int(len(expected_indices)),
        "output_sha256": sha256(paths["output"]),
    }
    mismatches = {
        key: (acceptance.get(key), value)
        for key, value in expected_contract.items()
        if acceptance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"QM9 RWSE acceptance mismatch: {mismatches}")
    encoded = torch.load(paths["output"], map_location="cpu", weights_only=False)
    if len(encoded) != len(expected_indices):
        raise ValueError("QM9 RWSE accepted cache has the wrong row count")
    positional_by_source = {}
    for graph in encoded:
        source_idx = int(graph.source_idx.view(-1)[0])
        positional = graph.random_walk_pe
        if positional.shape != (graph.num_nodes, walk_length):
            raise ValueError("QM9 RWSE accepted cache has a malformed row")
        if not torch.isfinite(positional).all():
            raise ValueError("QM9 RWSE accepted cache contains non-finite values")
        positional_by_source[source_idx] = positional
    if set(positional_by_source) != set(expected_indices.tolist()):
        raise ValueError("QM9 RWSE accepted cache has different source indices")
    for graphs in graph_splits.values():
        for graph in graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            positional = positional_by_source[source_idx]
            if positional.shape[0] != graph.num_nodes:
                raise ValueError("QM9 RWSE node count differs from topology graph")
            graph.random_walk_pe = positional.clone()
    return acceptance


def qm9_multihop_cache_paths(
    cache_dir: Path,
    split: ScreenSplit,
    *,
    max_distance: int = 4,
) -> dict[str, Path]:
    root = cache_dir / "structural" / (
        f"topology_{split.fingerprint}_multihop{max_distance}_v1"
    )
    return {
        "root": root,
        "parts": root / "parts",
        "progress": root / "progress.json",
        "acceptance": root / "multihop_acceptance.json",
    }


def _multihop_topology_payload(
    record: dict,
    source_idx: int,
    *,
    max_distance: int,
) -> dict:
    """Encode directed shortest-path virtual edges without copying labels."""
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    edge_index = record["edge_index"].long().cpu()
    edge_attr = record["edge_attr"].float().cpu()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Topology edge_index must have shape [2, E]")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("Topology edge_attr does not align with edge_index")
    if edge_attr.shape[1] != 4:
        raise ValueError("QM9 multihop cache requires four bond channels")
    num_nodes = int(record["x"].shape[0])
    adjacency = [set() for _ in range(num_nodes)]
    direct_features = {}
    for column in range(edge_index.shape[1]):
        source = int(edge_index[0, column])
        target = int(edge_index[1, column])
        adjacency[source].add(target)
        adjacency[target].add(source)
        direct_features[(source, target)] = edge_attr[column]

    sources, targets, features = [], [], []
    zero_bond = torch.zeros(4, dtype=torch.float32)
    for source in range(num_nodes):
        distance = [-1] * num_nodes
        distance[source] = 0
        queue = [source]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            if distance[current] >= max_distance:
                continue
            for target in sorted(adjacency[current]):
                if distance[target] >= 0:
                    continue
                distance[target] = distance[current] + 1
                queue.append(target)
        for target, path_length in enumerate(distance):
            if path_length <= 0 or path_length > max_distance:
                continue
            path_code = torch.zeros(max_distance, dtype=torch.float32)
            path_code[path_length - 1] = 1.0
            bond = direct_features.get((source, target), zero_bond)
            sources.append(source)
            targets.append(target)
            features.append(torch.cat((bond, path_code)))
    if not features:
        raise ValueError(f"QM9 graph {source_idx} has no usable multihop edges")
    return {
        "source_idx": int(source_idx),
        "num_nodes": num_nodes,
        "edge_index": torch.tensor([sources, targets], dtype=torch.long),
        "edge_attr": torch.stack(features).contiguous(),
    }


def _multihop_parts_sha256(parts: list[dict]) -> str:
    identity = [
        {"name": row["name"], "rows": row["rows"], "sha256": row["sha256"]}
        for row in parts
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_qm9_multihop_screen_cache(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int = 42,
    max_distance: int = 4,
    shard_size: int = 2_000,
    cache_dir: Path = DEFAULT_CACHE,
    source_cache_dir: Path | None = None,
) -> dict:
    """Build a resumable train/validation-only shortest-path edge cache."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    records = load_qm9_records(source_cache_dir or cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    expected_indices = np.concatenate((split.train, split.validation)).astype(
        np.int64
    )
    paths = qm9_multihop_cache_paths(
        cache_dir, split, max_distance=max_distance
    )
    paths["parts"].mkdir(parents=True, exist_ok=True)
    expected_part_names = {
        f"part_{part_index:05d}.pt"
        for part_index, _ in enumerate(range(0, len(expected_indices), shard_size))
    }
    existing_part_names = {path.name for path in paths["parts"].glob("part_*.pt")}
    unexpected_parts = sorted(existing_part_names - expected_part_names)
    if unexpected_parts:
        raise ValueError(f"Unexpected stale multihop parts: {unexpected_parts}")

    part_records = []
    edge_rows = 0
    for part_index, start in enumerate(range(0, len(expected_indices), shard_size)):
        indices = expected_indices[start : start + shard_size]
        part_path = paths["parts"] / f"part_{part_index:05d}.pt"
        if part_path.is_file():
            payloads = torch.load(
                part_path, map_location="cpu", weights_only=False
            )
        else:
            payloads = [
                _multihop_topology_payload(
                    records[int(source_idx)],
                    int(source_idx),
                    max_distance=max_distance,
                )
                for source_idx in indices
            ]
            _atomic_torch_save(part_path, payloads)
        observed_indices = np.asarray(
            [int(payload["source_idx"]) for payload in payloads], dtype=np.int64
        )
        if not np.array_equal(observed_indices, indices):
            raise ValueError(f"Multihop part {part_path.name} has wrong indices")
        for payload in payloads:
            if payload["edge_index"].shape[0] != 2:
                raise ValueError("Malformed multihop edge_index")
            expected_width = 4 + max_distance
            if payload["edge_attr"].shape != (
                payload["edge_index"].shape[1],
                expected_width,
            ):
                raise ValueError("Malformed multihop edge_attr")
            if not torch.isfinite(payload["edge_attr"]).all():
                raise ValueError("Multihop edge_attr contains non-finite values")
            edge_rows += int(payload["edge_index"].shape[1])
        part_records.append(
            {
                "name": part_path.name,
                "rows": len(payloads),
                "bytes": part_path.stat().st_size,
                "sha256": sha256(part_path),
            }
        )
        _atomic_json(
            paths["progress"],
            {
                "format": "molgap-qm9-multihop-progress-v1",
                "complete_parts": part_index + 1,
                "total_parts": len(expected_part_names),
                "rows": sum(row["rows"] for row in part_records),
                "test_role_read": False,
            },
        )
    acceptance = {
        "format": "molgap-qm9-multihop-acceptance-v1",
        "complete": True,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "index_sha256": hashlib.sha256(expected_indices.tobytes()).hexdigest(),
        "roles": {"train": train_size, "validation": validation_size},
        "requested_test_rows": test_size,
        "test_role_read": False,
        "max_distance": max_distance,
        "bond_feature_dim": 4,
        "distance_feature_dim": max_distance,
        "edge_feature_dim": 4 + max_distance,
        "rows": len(expected_indices),
        "edge_rows": edge_rows,
        "parts": part_records,
        "parts_sha256": _multihop_parts_sha256(part_records),
    }
    _atomic_json(paths["acceptance"], acceptance)
    return acceptance


def attach_accepted_qm9_multihop(
    graph_splits: dict[str, list[Data]],
    *,
    cache_dir: Path,
    split: ScreenSplit,
    max_distance: int = 4,
) -> dict:
    """Attach accepted sparse paths and refuse any test-role construction."""
    unexpected_roles = set(graph_splits) - {"train", "validation"}
    if unexpected_roles:
        raise ValueError(
            f"Multihop validation cache cannot serve roles: {unexpected_roles}"
        )
    paths = qm9_multihop_cache_paths(
        cache_dir, split, max_distance=max_distance
    )
    if not paths["acceptance"].is_file():
        raise FileNotFoundError(
            "Accepted QM9 multihop cache is required; run the CPU prep first"
        )
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    expected_indices = np.concatenate((split.train, split.validation)).astype(
        np.int64
    )
    expected_contract = {
        "format": "molgap-qm9-multihop-acceptance-v1",
        "complete": True,
        "split_seed": split.seed,
        "split_fingerprint": split.fingerprint,
        "index_sha256": hashlib.sha256(expected_indices.tobytes()).hexdigest(),
        "roles": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
        },
        "test_role_read": False,
        "max_distance": max_distance,
        "edge_feature_dim": 4 + max_distance,
        "rows": int(len(expected_indices)),
    }
    mismatches = {
        key: (acceptance.get(key), value)
        for key, value in expected_contract.items()
        if acceptance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"QM9 multihop acceptance mismatch: {mismatches}")
    parts = acceptance.get("parts", [])
    if acceptance.get("parts_sha256") != _multihop_parts_sha256(parts):
        raise ValueError("QM9 multihop aggregate hash mismatch")
    payload_by_source = {}
    for part in parts:
        part_path = paths["parts"] / part["name"]
        if not part_path.is_file() or sha256(part_path) != part["sha256"]:
            raise ValueError(f"QM9 multihop part mismatch: {part['name']}")
        payloads = torch.load(part_path, map_location="cpu", weights_only=False)
        if len(payloads) != int(part["rows"]):
            raise ValueError(f"QM9 multihop row mismatch: {part['name']}")
        for payload in payloads:
            source_idx = int(payload["source_idx"])
            if source_idx in payload_by_source:
                raise ValueError("Duplicate QM9 multihop source index")
            payload_by_source[source_idx] = payload
    if set(payload_by_source) != set(expected_indices.tolist()):
        raise ValueError("QM9 multihop cache has different source indices")
    for graphs in graph_splits.values():
        for graph in graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            payload = payload_by_source[source_idx]
            if int(payload["num_nodes"]) != graph.num_nodes:
                raise ValueError("QM9 multihop node count differs from graph")
            edge_index = payload["edge_index"]
            edge_attr = payload["edge_attr"]
            if edge_attr.shape != (edge_index.shape[1], 4 + max_distance):
                raise ValueError("QM9 multihop feature shape is malformed")
            if not torch.isfinite(edge_attr).all():
                raise ValueError("QM9 multihop feature contains non-finite values")
            graph.edge_index = edge_index.clone()
            graph.edge_attr = edge_attr.clone()
    return acceptance


def attach_accepted_qm9_multihop_attention(
    graph_splits: dict[str, list[Data]],
    *,
    cache_dir: Path,
    split: ScreenSplit,
    max_distance: int = 4,
) -> dict:
    """Attach path pairs for attention while preserving real-bond messages."""
    original = [
        (graph, graph.edge_index, graph.edge_attr)
        for graphs in graph_splits.values()
        for graph in graphs
    ]
    acceptance = attach_accepted_qm9_multihop(
        graph_splits,
        cache_dir=cache_dir,
        split=split,
        max_distance=max_distance,
    )
    for graph, bond_edge_index, bond_edge_attr in original:
        path_code = graph.edge_attr[:, 4:]
        if path_code.shape[1] != max_distance:
            raise ValueError("QM9 multihop path code has the wrong width")
        if not torch.allclose(
            path_code.sum(dim=1), torch.ones(path_code.shape[0])
        ):
            raise ValueError("QM9 multihop path code is not one-hot")
        graph.multihop_edge_index = graph.edge_index.clone()
        graph.multihop_distance = path_code.argmax(dim=1).long() + 1
        graph.edge_index = bond_edge_index
        graph.edge_attr = bond_edge_attr
    return acceptance


def _pcqm_transfer_features(graph: Data) -> Data:
    """Map QM9's processed 11-wide features to the PCQM 18-wide GPS contract.

    PCQM's checkpoint uses its declared 15-element order followed by degree,
    formal charge, and aromaticity.  Keeping this order exact is required for
    a warm start: the source checkpoint copies the original six element
    channels into columns 0--5 and its three scalar channels into 15--17.
    """
    z = graph.z.view(-1).long()
    features = torch.zeros((len(z), 18), dtype=torch.float32)
    pcqm_atom_order = (6, 7, 8, 9, 16, 17, 15, 35, 14, 5, 34, 32, 33, 12, 1)
    for column, atomic_number in enumerate(pcqm_atom_order):
        features[:, column] = (z == atomic_number).float()
    degree = torch.bincount(
        graph.edge_index[0].long(), minlength=len(z)
    ).float()
    features[:, 15] = degree / 4.0
    aromatic = torch.zeros(len(z), dtype=torch.float32)
    if graph.edge_attr.numel() and graph.edge_attr.shape[1] >= 4:
        aromatic_edges = graph.edge_attr[:, 3] > 0.5
        if aromatic_edges.any():
            aromatic_nodes = graph.edge_index[:, aromatic_edges].reshape(-1)
            aromatic[aromatic_nodes.unique()] = 1.0
    features[:, 17] = aromatic
    graph.x = features
    return graph


def _remap_pcqm_transfer_graphs(graph_splits: dict[str, list[Data]]) -> None:
    for graphs in graph_splits.values():
        for graph in graphs:
            _pcqm_transfer_features(graph)


def make_encoder(candidate: str, in_channels: int = 11, edge_dim: int = 4):
    config = dict(ENCODER_CONFIGS[candidate])
    config.pop("batch_size")
    # Training-only switches are consumed by train_encoder, not model
    # constructors.  Keeping them in the config lets each architecture choose
    # a stable precision mode without changing its module signature.
    config.pop("amp", None)
    kind = config.pop("kind")
    config.pop("atom_geom_mode", None)
    config.pop("input_channels", None)
    edge_dim = int(config.pop("model_edge_dim", edge_dim))
    config.pop("multihop_max_distance", None)
    config.pop("multihop_attention_distance", None)
    consistent_head = bool(config.pop("consistent_head", False))
    if candidate == "gine6":
        return GINEWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "edge_global_2d":
        return EdgeGlobal2DWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "pair_triplet_2d":
        return PairTriplet2DWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "pair_triplet_2d_rich":
        return PairTriplet2DRichWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "pair_gps_2d":
        return PairGPS2DWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "pair_gps_2d_r2":
        return PairGPS2DR2Wrapper(
            in_channels=in_channels, edge_dim=edge_dim, **config
        ), kind
    if candidate.startswith("pair_gps_2d_r3_"):
        return PairGPS2DR3Wrapper(
            in_channels=in_channels,
            edge_dim=edge_dim,
            consistent_head=consistent_head,
            **config,
        ), kind
    if candidate in {
        "edge_state_structural_gps",
        "edge_state_structural_orbital",
        "edge_state_structural_readout",
        "edge_state_structural_jk_readout",
        "edge_conditioned_structural_gps",
        "graph_token_structural_gps",
        "multihop_edge_state_structural_gps",
        "sparse_path_attention_structural_gps",
        "directed_edge_state_structural_gps",
    }:
        model_classes = {
            "edge_state_structural_gps": EdgeStateStructuralGPSWrapper,
            "edge_state_structural_orbital": EdgeStateStructuralGPSWrapper,
            "edge_state_structural_readout": EdgeReadoutStructuralGPSWrapper,
            "edge_state_structural_jk_readout": EdgeJKReadoutStructuralGPSWrapper,
            "edge_conditioned_structural_gps": EdgeConditionedStructuralGPSWrapper,
            "graph_token_structural_gps": GraphTokenStructuralGPSWrapper,
            "multihop_edge_state_structural_gps": EdgeStateStructuralGPSWrapper,
            "sparse_path_attention_structural_gps": (
                SparsePathAttentionStructuralGPSWrapper
            ),
            "directed_edge_state_structural_gps": (
                DirectedEdgeStateStructuralGPSWrapper
            ),
        }
        model_class = model_classes[candidate]
        model = model_class(in_channels=in_channels, edge_dim=edge_dim, **config)
        if consistent_head:
            model.head = FrontierCenterGapHead(
                model.node_emb.out_features,
                hidden_channels=model.node_emb.out_features,
                dropout=float(ENCODER_CONFIGS[candidate]["dropout"]),
            )
        return model, kind
    if candidate == "tgt_egt_hybrid":
        return TGTEGTHybridWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate in {"tgt_egt_compact", "tgt_egt_stable"}:
        return TGTCompactEGTWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "tgt_egt_rich":
        return TGTEGTRichWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate in {"tgt_egt_hybrid_plus", "tgt_egt_hybrid_frozen"}:
        return TGTEGTHybridPlusWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate in {
        "tgt_egt_hybrid_warmblend",
        "tgt_egt_hybrid_warmblend_frozen",
    }:
        return TGTEGTHybridWarmBlendWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate.startswith("gps"):
        return GPSWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate.startswith("schnet"):
        return SchNetWrapper(**config, use_charges=False), kind
    if candidate == "tensornet":
        return TensorNetWrapper(**config, use_charges=False), kind
    if candidate == "egnn":
        return EGNNWrapper(**config), kind
    if candidate == "tgt_lite":
        return TGTLiteWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "tgt_hybrid":
        return TGTLiteHybridWrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    if candidate == "tgt_hybrid_v2":
        return TGTLiteHybridV2Wrapper(in_channels=in_channels, edge_dim=edge_dim, **config), kind
    raise ValueError(candidate)


def _forward(kind: str, model, batch):
    if kind == "topology":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    if kind == "structural_topology":
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    if kind == "structural_multihop":
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.multihop_edge_index,
            batch.multihop_distance,
        )
    if isinstance(model, (TGTEGTHybridWrapper, TGTCompactEGTWrapper, TGTEGTRichWrapper, TGTEGTHybridPlusWrapper, TGTEGTHybridWarmBlendWrapper)):
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.z,
            batch.pos,
            batch.batch,
            topology_x=getattr(batch, "topology_x", None),
            topology_edges=getattr(batch, "topology_edges", None),
            topology_edge_attr=getattr(batch, "topology_edge_attr", None),
            topology_node_counts=getattr(batch, "topology_node_count", None),
            topology_edge_counts=getattr(batch, "topology_edge_count", None),
            geometry_node_counts=getattr(batch, "geometry_node_count", None),
        )
    if isinstance(model, (TGTLiteWrapper, TGTLiteHybridWrapper, TGTLiteHybridV2Wrapper)):
        topology_kwargs = {}
        if isinstance(model, TGTLiteHybridV2Wrapper):
            topology_kwargs = {
                "topology_x": getattr(batch, "topology_x", None),
                "topology_edges": getattr(batch, "topology_edges", None),
                "topology_edge_attr": getattr(batch, "topology_edge_attr", None),
                "topology_node_counts": getattr(batch, "topology_node_count", None),
                "topology_edge_counts": getattr(batch, "topology_edge_count", None),
                "geometry_node_counts": getattr(batch, "geometry_node_count", None),
            }
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.z,
            batch.pos,
            batch.batch,
            **topology_kwargs,
        )
    return model(
        batch.z,
        batch.pos,
        batch.batch,
        atom_geom=getattr(batch, "atom_geom", None),
    )


def _encode(kind: str, model, batch):
    if kind == "topology":
        return model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    if kind == "structural_topology":
        return model.encode(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    if kind == "structural_multihop":
        return model.encode(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.multihop_edge_index,
            batch.multihop_distance,
        )
    if isinstance(model, (TGTEGTHybridWrapper, TGTCompactEGTWrapper, TGTEGTRichWrapper, TGTEGTHybridPlusWrapper, TGTEGTHybridWarmBlendWrapper)):
        return model.encode(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.z,
            batch.pos,
            batch.batch,
            topology_x=getattr(batch, "topology_x", None),
            topology_edges=getattr(batch, "topology_edges", None),
            topology_edge_attr=getattr(batch, "topology_edge_attr", None),
            topology_node_counts=getattr(batch, "topology_node_count", None),
            topology_edge_counts=getattr(batch, "topology_edge_count", None),
            geometry_node_counts=getattr(batch, "geometry_node_count", None),
        )
    if isinstance(model, (TGTLiteWrapper, TGTLiteHybridWrapper, TGTLiteHybridV2Wrapper)):
        topology_kwargs = {}
        if isinstance(model, TGTLiteHybridV2Wrapper):
            topology_kwargs = {
                "topology_x": getattr(batch, "topology_x", None),
                "topology_edges": getattr(batch, "topology_edges", None),
                "topology_edge_attr": getattr(batch, "topology_edge_attr", None),
                "topology_node_counts": getattr(batch, "topology_node_count", None),
                "topology_edge_counts": getattr(batch, "topology_edge_count", None),
                "geometry_node_counts": getattr(batch, "geometry_node_count", None),
            }
        return model.encode(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.z,
            batch.pos,
            batch.batch,
            **topology_kwargs,
        )
    return model.encode(
        batch.z,
        batch.pos,
        batch.batch,
        atom_geom=getattr(batch, "atom_geom", None),
    )


def attach_local_geometry_features(
    graph_splits: dict[str, list[Data]],
    *,
    mode: str,
    cache_path: Path,
) -> dict:
    """Attach cached invariant atom features to geometry graph splits."""
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_status = "reused"
    else:
        cached = {}
        for graphs in graph_splits.values():
            for graph in graphs:
                source_idx = int(graph.source_idx.view(-1)[0])
                cached[source_idx] = local_geometry_features(graph.z, graph.pos)
        _atomic_torch_save(cache_path, cached)
        cache_status = "built"

    for graphs in graph_splits.values():
        for graph in graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            graph.atom_geom = select_geometry_features(cached[source_idx], mode)
    return {
        "mode": mode,
        "feature_dim": int(next(iter(graph_splits["train"])).atom_geom.shape[1]),
        "cache": str(cache_path),
        "cache_status": cache_status,
        "rows": len(cached),
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    errors = np.abs(prediction - target)
    result = {
        name: {"mae": float(errors[:, i].mean())}
        for i, name in enumerate(TARGET_NAMES)
    }
    result["average"] = {"mae": float(errors.mean())}
    return result


@torch.no_grad()
def evaluate_encoder(kind, model, graphs, batch_size, device, mean, std):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    predictions, targets, embeddings, source_indices = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        embedding = _encode(kind, model, batch)
        normalized = model.head(embedding)
        predictions.append((normalized * std.to(device) + mean.to(device)).float().cpu())
        targets.append(batch.y_eV.view(-1, 3).float().cpu())
        embeddings.append(embedding.float().cpu())
        source_indices.append(batch.source_idx.view(-1).cpu())
    return {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "embeddings": torch.cat(embeddings),
        "source_idx": torch.cat(source_indices),
    }


def train_encoder(
    *,
    candidate: str,
    geometry: str,
    train_size: int,
    validation_size: int,
    test_size: int,
    epochs: int,
    seed: int = 42,
    split_seed: int = 42,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    resume: bool = False,
    init_checkpoint: Path | None = None,
    expert_init_checkpoint: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    results_dir: Path = DEFAULT_RESULTS,
    models_dir: Path = DEFAULT_MODELS,
    embeddings_dir: Path | None = None,
    evaluate_test: bool = True,
    multihop_cache_dir: Path | None = None,
) -> dict:
    expected = ENCODER_CONFIGS[candidate]["kind"]
    structural_kinds = {"structural_topology", "structural_multihop"}
    if expected in {"topology", *structural_kinds} and geometry != "topology":
        raise ValueError(f"{candidate} requires --geometry topology")
    if expected == "geometry" and geometry not in {"dft", "etkdg"}:
        raise ValueError(f"{candidate} requires --geometry dft or etkdg")
    if not evaluate_test and expected not in structural_kinds:
        raise ValueError(
            "validation-only mode is implemented only for accepted structural caches"
        )
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(len(records), train_size, validation_size, test_size, split_seed)
    mean, std = target_stats(records, split.train)
    if expected in structural_kinds:
        graph_splits = {
            "train": [
                _topology_graph(records[int(i)], int(i), mean, std)
                for i in split.train
            ],
            "validation": [
                _topology_graph(records[int(i)], int(i), mean, std)
                for i in split.validation
            ],
        }
        geometry_report = {
            "geometry": "topology",
            "failed": 0,
            "test_role_read_during_selection": False,
        }
        rwse_dim = int(ENCODER_CONFIGS[candidate]["rwse_dim"])
        geometry_report["rwse_cache"] = attach_accepted_qm9_rwse(
            graph_splits,
            cache_dir=cache_dir,
            split=split,
            walk_length=rwse_dim,
        )
        multihop_max_distance = ENCODER_CONFIGS[candidate].get(
            "multihop_max_distance"
        )
        if multihop_max_distance is not None:
            if multihop_cache_dir is None:
                raise ValueError(f"{candidate} requires an accepted multihop cache")
            geometry_report["multihop_cache"] = attach_accepted_qm9_multihop(
                graph_splits,
                cache_dir=multihop_cache_dir,
                split=split,
                max_distance=int(multihop_max_distance),
            )
        multihop_attention_distance = ENCODER_CONFIGS[candidate].get(
            "multihop_attention_distance"
        )
        if multihop_attention_distance is not None:
            if multihop_cache_dir is None:
                raise ValueError(f"{candidate} requires an accepted multihop cache")
            geometry_report["multihop_attention_cache"] = (
                attach_accepted_qm9_multihop_attention(
                    graph_splits,
                    cache_dir=multihop_cache_dir,
                    split=split,
                    max_distance=int(multihop_attention_distance),
                )
            )
    else:
        graph_splits, geometry_report = make_graph_splits(
            records, split, geometry, mean, std, cache_dir, seed
        )
    if candidate == "gps9_pcqm_transfer":
        _remap_pcqm_transfer_graphs(graph_splits)
        geometry_report["node_feature_adapter"] = "pcqm_18w_v2_exact_atom_order"
    atom_geom_mode = ENCODER_CONFIGS[candidate].get("atom_geom_mode")
    if atom_geom_mode:
        feature_cache = (
            cache_dir
            / "geometry_features"
            / f"{geometry}_{split.fingerprint}_seed{seed}_v1.pt"
        )
        geometry_report["local_geometry_features"] = attach_local_geometry_features(
            graph_splits,
            mode=atom_geom_mode,
            cache_path=feature_cache,
        )
    input_channels = int(ENCODER_CONFIGS[candidate].get("input_channels", 11))
    model, kind = make_encoder(candidate, in_channels=input_channels)
    frontier_head_report = configure_frontier_head(
        model, records, split.train, mean, std
    )
    if init_checkpoint is not None:
        loaded = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        state = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
        if candidate in {
            "tgt_egt_hybrid_plus",
            "tgt_egt_hybrid_frozen",
            "tgt_egt_hybrid_warmblend",
            "tgt_egt_hybrid_warmblend_frozen",
        }:
            state = {f"base.{key}": value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"initialized {candidate} from {init_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    if expert_init_checkpoint is not None:
        loaded_expert = torch.load(
            expert_init_checkpoint, map_location="cpu", weights_only=True
        )
        expert_state = (
            loaded_expert.get("model", loaded_expert)
            if isinstance(loaded_expert, dict)
            else loaded_expert
        )
        if candidate not in {
            "tgt_egt_hybrid_warmblend",
            "tgt_egt_hybrid_warmblend_frozen",
        }:
            raise ValueError(
                "expert_init_checkpoint is only supported by "
                "tgt_egt_hybrid_warmblend"
            )
        expert_state = {
            f"expert.{key}": value for key, value in expert_state.items()
        }
        missing, unexpected = model.load_state_dict(expert_state, strict=False)
        print(
            f"initialized {candidate} expert from {expert_init_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model = model.to(device)
    batch_size = int(ENCODER_CONFIGS[candidate]["batch_size"])
    train_loader = DataLoader(
        graph_splits["train"], batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        graph_splits["validation"], batch_size=batch_size, shuffle=False, num_workers=0
    )
    effective_learning_rate = float(
        ENCODER_CONFIGS[candidate].get("learning_rate", learning_rate)
    )
    effective_weight_decay = float(
        ENCODER_CONFIGS[candidate].get("weight_decay", weight_decay)
    )
    use_amp = device.type == "cuda" and bool(
        ENCODER_CONFIGS[candidate].get("amp", True)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_learning_rate,
        weight_decay=effective_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.L1Loss()
    run_name = (
        f"n{train_size}_{validation_size}_{test_size}/"
        f"{candidate}_{geometry}/seed{seed}"
    )
    result_path = results_dir / run_name / "metrics.json"
    effective_embeddings_dir = embeddings_dir or (cache_dir / "embeddings")
    embedding_path = effective_embeddings_dir / run_name / "payload.pt"
    model_path = models_dir / run_name / "model.pt"
    checkpoint_path = models_dir / run_name / "checkpoint.pt"
    best_mae = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    log = []
    if init_checkpoint is not None and (
        not resume or not checkpoint_path.exists()
    ):
        initial_validation = evaluate_encoder(
            kind,
            model,
            graph_splits["validation"],
            batch_size,
            device,
            mean,
            std,
        )
        initial_metrics = _metrics(
            initial_validation["predictions"].numpy(),
            initial_validation["targets"].numpy(),
        )
        best_mae = float(initial_metrics["average"]["mae"])
        best_state = copy.deepcopy(model.state_dict())
        print(
            f"warm-start validation={best_mae:.5f}eV; "
            "retained as initial best checkpoint",
            flush=True,
        )
    start_epoch = 0
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_mae = float(checkpoint["best_mae"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(_forward(kind, model, batch), batch.y.view(-1, 3))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
        scheduler.step()
        validation = evaluate_encoder(
            kind, model, graph_splits["validation"], batch_size, device, mean, std
        )
        metrics = _metrics(
            validation["predictions"].numpy(), validation["targets"].numpy()
        )
        val_mae = metrics["average"]["mae"]
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_normalized_l1": total / max(count, 1),
            "validation_average_mae_eV": val_mae,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_state": best_state,
            "best_mae": best_mae,
            "best_epoch": best_epoch,
            "wait": wait,
            "log": log,
        }, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        print(
            f"{candidate}/{geometry} ep{epoch:02d} "
            f"train={row['train_normalized_l1']:.5f} "
            f"val={val_mae:.5f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    if expected in structural_kinds and evaluate_test:
        if (
            ENCODER_CONFIGS[candidate].get("multihop_max_distance") is not None
            or ENCODER_CONFIGS[candidate].get("multihop_attention_distance")
            is not None
        ):
            raise ValueError(
                "Multihop test construction requires a separately authorized cache"
            )
        graph_splits["test"] = [
            _topology_graph(records[int(i)], int(i), mean, std)
            for i in split.test
        ]
        attach_accepted_qm9_rwse(
            {"test": graph_splits["test"]},
            cache_dir=cache_dir,
            split=split,
            walk_length=int(ENCODER_CONFIGS[candidate]["rwse_dim"]),
        )
        geometry_report["test_role_read_after_selection"] = True
    elif expected in structural_kinds:
        geometry_report["test_role_read_after_selection"] = False

    role_payloads = {}
    role_metrics = {}
    for role, graphs in graph_splits.items():
        payload = evaluate_encoder(kind, model, graphs, batch_size, device, mean, std)
        role_payloads[role] = payload
        role_metrics[role] = _metrics(
            payload["predictions"].numpy(), payload["targets"].numpy()
        )

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(embedding_path, role_payloads)
    _atomic_torch_save(model_path, best_state)
    result = {
        "experiment": "qm9_architecture_screen",
        "candidate": candidate,
        "geometry": geometry,
        "seed": seed,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "split_rows": {role: len(graphs) for role, graphs in graph_splits.items()},
        "requested_rows": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "test_role_evaluated": "test" in graph_splits,
        "target_names": list(TARGET_NAMES),
        "target_units": "eV",
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "model_config": ENCODER_CONFIGS[candidate],
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "metrics": role_metrics,
        "geometry_report": geometry_report,
        "frontier_head": frontier_head_report,
        "log": log,
        "artifacts": {
            "embeddings": str(embedding_path),
            "model": str(model_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    _atomic_json(result_path, result)
    return result


def evaluate_structural_checkpoint_test(
    *,
    candidate: str,
    checkpoint: Path,
    output: Path,
    payload_output: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int = 42,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    """Evaluate one validation-selected structural checkpoint on QM9 test once."""
    if ENCODER_CONFIGS[candidate]["kind"] != "structural_topology":
        raise ValueError(f"{candidate} is not a structural topology encoder")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    test_graphs = [
        _topology_graph(records[int(i)], int(i), mean, std)
        for i in split.test
    ]
    acceptance = attach_accepted_qm9_rwse(
        {"test": test_graphs},
        cache_dir=cache_dir,
        split=split,
        walk_length=int(ENCODER_CONFIGS[candidate]["rwse_dim"]),
    )
    input_channels = int(ENCODER_CONFIGS[candidate].get("input_channels", 11))
    model, kind = make_encoder(candidate, in_channels=input_channels)
    frontier_head_report = configure_frontier_head(
        model, records, split.train, mean, std
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_size = int(ENCODER_CONFIGS[candidate]["batch_size"])
    payload = evaluate_encoder(
        kind, model, test_graphs, batch_size, device, mean, std
    )
    metrics = _metrics(
        payload["predictions"].numpy(), payload["targets"].numpy()
    )
    _atomic_torch_save(payload_output, {"test": payload})
    result = {
        "experiment": "qm9_structural_single_test",
        "candidate": candidate,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "test_rows": len(test_graphs),
        "test_role_read_once": True,
        "rwse_output_sha256": acceptance["output_sha256"],
        "frontier_head": frontier_head_report,
        "metrics": metrics,
        "payload": str(payload_output),
    }
    _atomic_json(output, result)
    return result


def evaluate_encoder_on_geometry(
    *,
    candidate: str,
    geometry: str,
    checkpoint: Path,
    output: Path,
    embedding_output: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
    geometry_seed: int,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    if ENCODER_CONFIGS[candidate]["kind"] != "geometry":
        raise ValueError(f"{candidate} is not a geometry encoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    graph_splits, geometry_report = make_graph_splits(
        records,
        split,
        geometry,
        mean,
        std,
        cache_dir,
        geometry_seed,
    )
    model, kind = make_encoder(candidate)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    batch_size = int(ENCODER_CONFIGS[candidate]["batch_size"])
    role_payloads = {
        role: evaluate_encoder(
            kind, model, graphs, batch_size, device, mean, std
        )
        for role, graphs in graph_splits.items()
    }
    role_metrics = {
        role: _metrics(
            payload["predictions"].numpy(), payload["targets"].numpy()
        )
        for role, payload in role_payloads.items()
    }
    _atomic_torch_save(embedding_output, role_payloads)
    result = {
        "experiment": "qm9_geometry_transfer_eval",
        "candidate": candidate,
        "geometry": geometry,
        "geometry_seed": geometry_seed,
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "checkpoint": str(checkpoint),
        "embedding_output": str(embedding_output),
        "split_rows": {
            role: len(graphs) for role, graphs in graph_splits.items()
        },
        "metrics": role_metrics,
        "geometry_report": geometry_report,
    }
    _atomic_json(output, result)
    return result


@torch.no_grad()
def export_gps_multiscale_embeddings(
    *,
    checkpoint: Path,
    output: Path,
    embedding_output: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
    layers: tuple[int, ...] = (2, 4, -1),
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    """Export several GPS9 layers without adding another encoder forward pass."""
    if -1 not in layers:
        raise ValueError("GPS multiscale export requires -1 for final predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_qm9_records(cache_dir)
    split = fixed_split(
        len(records), train_size, validation_size, test_size, split_seed
    )
    mean, std = target_stats(records, split.train)
    graph_splits, _ = make_graph_splits(
        records, split, "topology", mean, std, cache_dir, seed=split_seed
    )
    model, _ = make_encoder("gps9")
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device)
    model.eval()
    batch_size = int(ENCODER_CONFIGS["gps9"]["batch_size"])
    hidden = int(ENCODER_CONFIGS["gps9"]["hidden_channels"])
    role_payloads = {}
    for role, graphs in graph_splits.items():
        loader = DataLoader(
            graphs, batch_size=batch_size, shuffle=False, num_workers=0
        )
        predictions, targets, embeddings, source_indices = [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            embedding = model.encode_layers(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                layers=layers,
            )
            normalized = model.head(embedding[:, -hidden:])
            predictions.append(
                (normalized * std.to(device) + mean.to(device)).float().cpu()
            )
            targets.append(batch.y_eV.view(-1, 3).float().cpu())
            embeddings.append(embedding.float().cpu())
            source_indices.append(batch.source_idx.view(-1).cpu())
        role_payloads[role] = {
            "predictions": torch.cat(predictions),
            "targets": torch.cat(targets),
            "embeddings": torch.cat(embeddings),
            "source_idx": torch.cat(source_indices),
        }
    _atomic_torch_save(embedding_output, role_payloads)
    result = {
        "experiment": "qm9_gps_multiscale_export",
        "candidate": "gps9",
        "layers": list(layers),
        "split_seed": split_seed,
        "split_fingerprint": split.fingerprint,
        "checkpoint": str(checkpoint),
        "embedding_output": str(embedding_output),
        "embedding_dim": int(role_payloads["train"]["embeddings"].shape[1]),
        "split_rows": {
            role: len(payload["source_idx"])
            for role, payload in role_payloads.items()
        },
        "metrics": {
            role: _metrics(
                payload["predictions"].numpy(), payload["targets"].numpy()
            )
            for role, payload in role_payloads.items()
        },
    }
    _atomic_json(output, result)
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
