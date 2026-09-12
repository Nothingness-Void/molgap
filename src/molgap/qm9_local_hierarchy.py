"""QM9-30K local-hierarchy pretraining screen for the Track C funnel."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np


TRAIN_ROWS = 30_000
VALIDATION_ROWS = 3_000
HELD_OUT_ROWS = 3_000
SPLIT_SEED = 42
MODEL_SEED = 42
RWSE_DIM = 16
BATCH_SIZE = 48
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
SCRATCH_EPOCHS = 40
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20
PATIENCE = 8
MASK_RATE = 0.15
MIN_GAIN_EV = 0.002
EXPECTED_SOURCE_ROWS = 130_831
EXPECTED_RDKIT_VERSION = "2023.09.6"
EXPECTED_PROCESSED_SHA256 = (
    "90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721"
)
EXPECTED_RAW_SDF_SHA256 = (
    "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
)

FUNCTIONAL_GROUP_SMARTS = {
    "aromatic": "[a]",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "carboxyl": "[CX3](=[OX1])[OX2H1]",
    "nitrile": "[CX2]#N",
    "amine": "[NX3;H0,H1,H2;!$(NC=O)]",
    "alcohol": "[OX2H][#6]",
    "ether": "[OD2]([#6])[#6]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "ring_heteroatom": "[n,o,s;r]",
}


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


def _smarts_sha256() -> str:
    payload = json.dumps(
        FUNCTIONAL_GROUP_SMARTS, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _functional_group_labels(molecule):
    import torch
    from rdkit import Chem

    labels = torch.zeros(
        (molecule.GetNumAtoms(), len(FUNCTIONAL_GROUP_SMARTS)),
        dtype=torch.float32,
    )
    for column, smarts in enumerate(FUNCTIONAL_GROUP_SMARTS.values()):
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise RuntimeError(f"Invalid frozen SMARTS: {smarts}")
        for match in molecule.GetSubstructMatches(query):
            labels[list(match), column] = 1.0
    return labels


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
        raise ValueError("Canonical QM9 SMILES did not sanitize")
    return molecule, smiles


def _canonical_valid_pool(records: list[dict], supplier):
    """Freeze the source rows that survive the required canonical round trip."""
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
            valid.append(source_idx)
    return np.asarray(valid, dtype=np.int64), invalid


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def _make_graph(record: dict, source_idx: int, supplier):
    import torch
    from ogb.utils.mol import smiles2graph
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddRandomWalkPE

    molecule, smiles = _record_molecule(record, supplier)
    payload = smiles2graph(smiles)
    graph = Data(
        x=torch.as_tensor(payload["node_feat"], dtype=torch.long),
        edge_index=torch.as_tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(payload["edge_feat"], dtype=torch.long),
        y=record["y"].view(-1)[4].float().view(1),
        row_id=torch.tensor([int(source_idx)], dtype=torch.long),
        functional_group_y=_functional_group_labels(molecule),
    )
    if graph.x.shape != (molecule.GetNumAtoms(), 9):
        raise RuntimeError("OGB atom representation is not aligned to QM9")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
        raise RuntimeError("OGB bond representation changed")
    return AddRandomWalkPE(
        walk_length=RWSE_DIM, attr_name="random_walk_pe"
    )(graph)


def build_cache(output_root: Path, *, source_commit: str, shard_size: int = 2_000):
    """Build an immutable train/validation-only OGB/RWSE/SMARTS cache."""
    import torch
    from rdkit import Chem, RDLogger

    from rdkit import rdBase

    from .qm9_data import (
        fixed_split_from_pool,
        load_qm9_records,
        prepare_qm9_files,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final cache manifest exists")

    records = load_qm9_records(output_root / "source")
    files = prepare_qm9_files(output_root / "source")
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
    valid_pool, canonical_invalid = _canonical_valid_pool(records, supplier)
    split = fixed_split_from_pool(
        valid_pool, TRAIN_ROWS, VALIDATION_ROWS, HELD_OUT_ROWS, SPLIT_SEED
    )
    validity = {
        "format": "molgap-qm9-canonical-valid-pool-v1",
        "source_commit": source_commit,
        "rdkit_version": rdBase.rdkitVersion,
        "total_source_rows": len(records),
        "valid_source_rows": int(valid_pool.size),
        "invalid_source_rows": len(canonical_invalid),
        "valid_source_indices": valid_pool.tolist(),
        "valid_source_indices_sha256": _indices_sha256(valid_pool),
        "invalid_records": canonical_invalid,
        "processed_source_sha256": sha256_file(files["processed"]),
        "raw_sdf_sha256": sha256_file(files["raw_sdf"]),
        "selected_train_indices": split.train.tolist(),
        "selected_validation_indices": split.validation.tolist(),
        "selected_split_fingerprint": split.fingerprint,
        "held_out_graphs_materialized": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "canonical_validity.json", validity)
    roles = {"train": split.train, "validation": split.validation}
    shards = []
    failures = []
    group_counts = torch.zeros(len(FUNCTIONAL_GROUP_SMARTS), dtype=torch.long)
    for role, indices in roles.items():
        for part_number, start in enumerate(range(0, len(indices), shard_size)):
            stop = min(start + shard_size, len(indices))
            part_path = output_root / f"{role}_part_{part_number:03d}.pt"
            expected = np.asarray(indices[start:stop], dtype=np.int64)
            if part_path.is_file():
                graphs = torch.load(
                    part_path, map_location="cpu", weights_only=False
                )
            else:
                graphs = []
                for source_idx in expected:
                    try:
                        graph = _make_graph(
                            records[int(source_idx)], int(source_idx), supplier
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "role": role,
                                "source_idx": int(source_idx),
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        raise RuntimeError(
                            f"QM9 graph construction failed at {source_idx}"
                        ) from error
                    graphs.append(graph)
                atomic_torch_save(part_path, graphs)
            for graph in graphs:
                group_counts += graph.functional_group_y.sum(dim=0).long()
            observed = np.asarray(
                [int(graph.row_id.view(-1)[0]) for graph in graphs],
                dtype=np.int64,
            )
            if not np.array_equal(observed, expected):
                raise RuntimeError(f"Cache identity changed for {part_path.name}")
            shards.append(
                {
                    "role": role,
                    "file": part_path.name,
                    "graph_count": len(graphs),
                    "sha256": sha256_file(part_path),
                }
            )
            atomic_json(
                output_root / "progress.json",
                {
                    "format": "molgap-qm9-local-hierarchy-cache-progress-v1",
                    "source_commit": source_commit,
                    "split_fingerprint": split.fingerprint,
                    "completed_shards": shards,
                    "failures": failures,
                    "test_role_read": False,
                },
            )

    aggregate = hashlib.sha256()
    for shard in shards:
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    manifest = {
        "format": "molgap-qm9-edgestate-local-hierarchy-cache-v2",
        "complete": True,
        "source_commit": source_commit,
        "split_seed": SPLIT_SEED,
        "split_fingerprint": split.fingerprint,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "held_out_indices_materialized": False,
        "canonical_valid_pool_count": int(valid_pool.size),
        "canonical_invalid_count": len(canonical_invalid),
        "canonical_valid_pool_sha256": _indices_sha256(valid_pool),
        "canonical_validity_sha256": sha256_file(
            output_root / "canonical_validity.json"
        ),
        "train_source_indices_sha256": _indices_sha256(split.train),
        "validation_source_indices_sha256": _indices_sha256(split.validation),
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": RWSE_DIM,
        "functional_groups": list(FUNCTIONAL_GROUP_SMARTS),
        "functional_group_smarts_sha256": _smarts_sha256(),
        "functional_group_positive_counts": group_counts.tolist(),
        "failures": failures,
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "gpu_used": False,
        "model_inference_executed": False,
        "test_role_read": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_cache(cache_root: Path, expected_sha256: str | None = None):
    import torch

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-edgestate-local-hierarchy-cache-v2",
        "complete": True,
        "roles": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS},
        "held_out_indices_materialized": False,
        "atom_feature_channels": 9,
        "bond_feature_channels": 3,
        "rwse_channels": RWSE_DIM,
        "functional_group_smarts_sha256": _smarts_sha256(),
        "test_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Cache contract changed for {key}")
    validity_path = cache_root / "canonical_validity.json"
    if not validity_path.is_file():
        raise RuntimeError("Canonical-valid pool evidence is missing")
    if sha256_file(validity_path) != manifest.get("canonical_validity_sha256"):
        raise RuntimeError("Canonical-valid pool evidence hash changed")
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    valid_pool = np.asarray(validity.get("valid_source_indices", []), dtype=np.int64)
    train_indices = np.asarray(
        validity.get("selected_train_indices", []), dtype=np.int64
    )
    validation_indices = np.asarray(
        validity.get("selected_validation_indices", []), dtype=np.int64
    )
    if (
        validity.get("format") != "molgap-qm9-canonical-valid-pool-v1"
        or validity.get("total_source_rows") != EXPECTED_SOURCE_ROWS
        or validity.get("rdkit_version") != EXPECTED_RDKIT_VERSION
        or validity.get("processed_source_sha256") != EXPECTED_PROCESSED_SHA256
        or validity.get("raw_sdf_sha256") != EXPECTED_RAW_SDF_SHA256
        or validity.get("held_out_graphs_materialized") is not False
        or validity.get("test_role_read") is not False
        or _indices_sha256(valid_pool)
        != manifest.get("canonical_valid_pool_sha256")
        or int(valid_pool.size) != manifest.get("canonical_valid_pool_count")
        or len(validity.get("invalid_records", []))
        != manifest.get("canonical_invalid_count")
        or _indices_sha256(train_indices)
        != manifest.get("train_source_indices_sha256")
        or _indices_sha256(validation_indices)
        != manifest.get("validation_source_indices_sha256")
        or np.intersect1d(train_indices, validation_indices).size
        or not np.isin(train_indices, valid_pool).all()
        or not np.isin(validation_indices, valid_pool).all()
    ):
        raise RuntimeError("Canonical-valid pool contract changed")
    acceptance_path = cache_root / "acceptance.json"
    if not acceptance_path.is_file():
        raise RuntimeError("Independent cache acceptance is missing")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    for key, value in {
        "format": "molgap-qm9-edgestate-local-hierarchy-cache-acceptance-v2",
        "accepted": True,
        "source_commit": manifest["source_commit"],
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "canonical_validity_sha256": manifest["canonical_validity_sha256"],
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }.items():
        if acceptance.get(key) != value:
            raise RuntimeError(f"Cache acceptance changed for {key}")
    if expected_sha256 and manifest.get("aggregate_sha256") != expected_sha256:
        raise RuntimeError("Cache aggregate SHA changed")
    roles = {"train": [], "validation": []}
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = cache_root / shard["file"]
        observed_sha = sha256_file(path)
        if observed_sha != shard["sha256"]:
            raise RuntimeError(f"Cache shard changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{observed_sha}\n".encode("ascii")
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload) != shard["graph_count"]:
            raise RuntimeError(f"Cache shard count changed: {path.name}")
        roles[shard["role"]].extend(payload)
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Cache aggregate hash changed")
    if {key: len(value) for key, value in roles.items()} != required["roles"]:
        raise RuntimeError("Cache role count changed")
    return roles, manifest


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_encoder():
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
        rwse_dim=16,
        edge_state_channels=64,
    )


def forward_encoder(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


class LocalHierarchyHeads:
    """Training-only local reconstruction heads attached by forward hooks."""

    def __init__(self, model):
        import torch.nn as nn
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.node_state = None
        self.edge_state = None
        self.atom_heads = nn.ModuleList(
            nn.Linear(192, classes) for classes in get_atom_feature_dims()
        )
        self.bond_heads = nn.ModuleList(
            nn.Linear(64, classes) for classes in get_bond_feature_dims()
        )
        self.group_head = nn.Linear(192, len(FUNCTIONAL_GROUP_SMARTS))
        self.module = nn.ModuleDict(
            {
                "atom_heads": self.atom_heads,
                "bond_heads": self.bond_heads,
                "group_head": self.group_head,
            }
        )
        self.handles = [
            model.convs[-1].register_forward_hook(self._capture_node),
            model.edge_updates[-1].register_forward_hook(self._capture_edge),
        ]

    def _capture_node(self, _module, _inputs, output):
        self.node_state = output

    def _capture_edge(self, _module, _inputs, output):
        self.edge_state = output

    def close(self):
        for handle in self.handles:
            handle.remove()


def _make_loader(graphs, *, shuffle: bool, seed: int):
    import torch
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
    import torch

    targets = torch.stack([graph.y.view(()) for graph in graphs])
    return targets.mean(), targets.std().clamp_min(1e-6)


def _evaluate(model, loader, mean, std, device):
    import torch

    model.eval()
    absolute = 0.0
    rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = forward_encoder(model, batch) * std + mean
            absolute += (prediction - batch.y.view(-1)).abs().sum().item()
            rows += int(prediction.numel())
    return absolute / rows


def _state_sha256(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_state(loader, mask_generator=None):
    import torch

    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": loader.generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    if mask_generator is not None:
        state["mask"] = mask_generator.get_state()
    return state


def _restore_rng(state, loader, mask_generator=None):
    import torch

    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    loader.generator.set_state(state["loader"])
    torch.cuda.set_rng_state_all(state["cuda"])
    if mask_generator is not None:
        mask_generator.set_state(state["mask"])


def train_gap(
    model,
    roles,
    output_dir: Path,
    *,
    epochs: int,
    seed: int,
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
        optimizer, T_max=epochs, eta_min=1e-6
    )
    train_loader = _make_loader(roles["train"], shuffle=True, seed=seed)
    validation_loader = _make_loader(
        roles["validation"], shuffle=False, seed=seed
    )
    trace = []
    best = float("inf")
    stale = 0
    best_epoch = -1
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("source_commit") != source_commit
            or checkpoint.get("cache_sha256") != cache_sha256
            or checkpoint.get("phase") != "gap"
            or checkpoint.get("max_epochs") != epochs
            or checkpoint.get("seed") != seed
        ):
            raise RuntimeError("Gap checkpoint contract changed")
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
            start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_encoder(model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += (prediction.detach() * std + mean - batch.y.view(-1)).abs().sum().item()
            rows += int(prediction.numel())
        validation_mae = _evaluate(model, validation_loader, mean, std, device)
        improved = validation_mae < best
        if improved:
            best = validation_mae
            best_epoch = epoch
            stale = 0
            atomic_torch_save(output_dir / "best_model.pt", model.state_dict())
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
            output_dir / "last_checkpoint.pt",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "phase": "gap",
                "max_epochs": epochs,
                "seed": seed,
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
    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
    )
    return model, {
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "model_sha256": sha256_file(output_dir / "best_model.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "last_checkpoint.pt"),
    }


def pretrain_local_hierarchy(
    model,
    roles,
    output_dir: Path,
    *,
    seed: int,
    source_commit: str,
    cache_sha256: str,
):
    import torch
    import torch.nn.functional as functional

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model.to(device)
    heads = LocalHierarchyHeads(model)
    heads.module = heads.module.to(device)
    parameters = list(model.parameters()) + list(heads.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
    )
    loader = _make_loader(roles["train"], shuffle=True, seed=seed)
    generator = torch.Generator().manual_seed(seed + 17)
    trace = []
    start_epoch = 0
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if (
            checkpoint.get("source_commit") != source_commit
            or checkpoint.get("cache_sha256") != cache_sha256
            or checkpoint.get("phase") != "local_hierarchy"
            or checkpoint.get("max_epochs") != PRETRAIN_EPOCHS
            or checkpoint.get("seed") != seed
        ):
            raise RuntimeError("Pretraining checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        heads.module.load_state_dict(checkpoint["heads"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng(checkpoint["rng"], loader, generator)
    for epoch in range(start_epoch, PRETRAIN_EPOCHS):
        model.train()
        heads.module.train()
        totals = {"loss": 0.0, "atom": 0.0, "bond": 0.0, "group": 0.0}
        batches = 0
        started = time.perf_counter()
        for batch in loader:
            original_x = batch.x.clone()
            original_edge = batch.edge_attr.clone()
            node_mask = torch.rand(batch.x.shape[0], generator=generator) < MASK_RATE
            edge_mask = torch.rand(batch.edge_attr.shape[0], generator=generator) < MASK_RATE
            if not bool(node_mask.any()):
                node_mask[0] = True
            if edge_mask.numel() and not bool(edge_mask.any()):
                edge_mask[0] = True
            batch.x[node_mask] = 0
            if edge_mask.numel():
                batch.edge_attr[edge_mask] = 0
            batch = batch.to(device)
            node_mask = node_mask.to(device)
            edge_mask = edge_mask.to(device)
            original_x = original_x.to(device)
            original_edge = original_edge.to(device)
            optimizer.zero_grad(set_to_none=True)
            forward_encoder(model, batch)
            atom_loss = sum(
                functional.cross_entropy(head(heads.node_state[node_mask]), original_x[node_mask, column])
                for column, head in enumerate(heads.atom_heads)
            ) / len(heads.atom_heads)
            if edge_mask.numel() and bool(edge_mask.any()):
                bond_loss = sum(
                    functional.cross_entropy(head(heads.edge_state[edge_mask]), original_edge[edge_mask, column])
                    for column, head in enumerate(heads.bond_heads)
                ) / len(heads.bond_heads)
            else:
                bond_loss = atom_loss.new_zeros(())
            group_loss = functional.binary_cross_entropy_with_logits(
                heads.group_head(heads.node_state[node_mask]),
                batch.functional_group_y[node_mask].float(),
            )
            loss = atom_loss + bond_loss + 0.5 * group_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for key, value in (
                ("loss", loss), ("atom", atom_loss),
                ("bond", bond_loss), ("group", group_loss),
            ):
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            **{key: value / batches for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_torch_save(
            output_dir / "last_checkpoint.pt",
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "heads": heads.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "cache_sha256": cache_sha256,
                "phase": "local_hierarchy",
                "max_epochs": PRETRAIN_EPOCHS,
                "seed": seed,
                "rng": _rng_state(loader, generator),
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"local_hierarchy ep{epoch:02d} loss={row['loss']:.6f} "
            f"atom={row['atom']:.6f} bond={row['bond']:.6f} "
            f"group={row['group']:.6f} {row['seconds']:.1f}s",
            flush=True,
        )
    heads.close()
    training_head_parameters = sum(p.numel() for p in heads.module.parameters())
    del heads
    return model, {
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "training_head_parameters": training_head_parameters,
        "checkpoint_sha256": sha256_file(output_dir / "last_checkpoint.pt"),
    }


def run_preflight(output_root: Path, *, source_commit: str):
    """Exercise the exact encoder and auxiliary heads on one synthetic DCU batch."""
    import torch
    import torch.nn.functional as functional
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("SCNet did not expose a DCU")
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(MODEL_SEED)
    graphs = []
    for graph_id in range(2):
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        graph = Data(
            x=torch.zeros((4, 9), dtype=torch.long),
            edge_index=edge_index,
            edge_attr=torch.zeros((6, 3), dtype=torch.long),
            y=torch.tensor([0.1 * graph_id], dtype=torch.float32),
            random_walk_pe=torch.zeros((4, RWSE_DIM), dtype=torch.float32),
            functional_group_y=torch.zeros(
                (4, len(FUNCTIONAL_GROUP_SMARTS)), dtype=torch.float32
            ),
        )
        graphs.append(graph)

    device = torch.device("cuda")
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False))).to(device)
    model = make_encoder().to(device)
    heads = LocalHierarchyHeads(model)
    heads.module = heads.module.to(device)
    prediction = forward_encoder(model, batch)
    atom_loss = sum(
        functional.cross_entropy(head(heads.node_state), batch.x[:, column])
        for column, head in enumerate(heads.atom_heads)
    ) / len(heads.atom_heads)
    bond_loss = sum(
        functional.cross_entropy(head(heads.edge_state), batch.edge_attr[:, column])
        for column, head in enumerate(heads.bond_heads)
    ) / len(heads.bond_heads)
    group_loss = functional.binary_cross_entropy_with_logits(
        heads.group_head(heads.node_state), batch.functional_group_y.float()
    )
    gap_loss = functional.l1_loss(prediction, batch.y.view(-1))
    loss = atom_loss + bond_loss + 0.5 * group_loss + gap_loss
    loss.backward()
    parameters = list(model.parameters()) + list(heads.module.parameters())
    finite = bool(torch.isfinite(loss).item()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )
    result = {
        "format": "molgap-qm9-edgestate-local-hierarchy-dcu-preflight-v2",
        "accepted": finite,
        "source_commit": source_commit,
        "architecture": "ogb_edge_state_structural_gps9",
        "gpu": torch.cuda.get_device_name(0),
        "cuda_available": True,
        "inference_parameter_count": sum(p.numel() for p in model.parameters()),
        "training_head_parameter_count": sum(
            p.numel() for p in heads.module.parameters()
        ),
        "node_state_shape": list(heads.node_state.shape),
        "edge_state_shape": list(heads.edge_state.shape),
        "finite_forward_backward": finite,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "model_inference_executed": True,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    heads.close()
    atomic_json(output_root / "preflight.json", result)
    if not finite:
        raise RuntimeError("DCU preflight produced non-finite values")
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

    if not torch.cuda.is_available():
        raise RuntimeError("SCNet did not expose a DCU")
    torch.backends.cuda.matmul.allow_tf32 = False
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    for key, value in {
        "format": "molgap-qm9-edgestate-local-hierarchy-dcu-preflight-v2",
        "accepted": True,
        "source_commit": source_commit,
        "architecture": "ogb_edge_state_structural_gps9",
        "cuda_available": True,
        "finite_forward_backward": True,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"DCU preflight contract changed for {key}")
    roles, manifest = load_cache(cache_root, cache_sha256)
    output_root.mkdir(parents=True, exist_ok=True)

    set_seed(MODEL_SEED)
    initial = make_encoder()
    initial_sha = _state_sha256(initial)
    inference_parameters = sum(parameter.numel() for parameter in initial.parameters())
    if inference_parameters != preflight.get("inference_parameter_count"):
        raise RuntimeError("Encoder parameter count changed after DCU preflight")
    results = {}
    for name in ("scratch_a", "scratch_b"):
        set_seed(MODEL_SEED)
        model = make_encoder()
        if _state_sha256(model) != initial_sha:
            raise RuntimeError("Scratch initialization changed")
        model, results[name] = train_gap(
            model,
            roles,
            output_root / name,
            epochs=SCRATCH_EPOCHS,
            seed=MODEL_SEED,
            source_commit=source_commit,
            cache_sha256=cache_sha256,
        )
        del model
        torch.cuda.empty_cache()

    set_seed(MODEL_SEED)
    candidate = make_encoder()
    if _state_sha256(candidate) != initial_sha:
        raise RuntimeError("Pretraining initialization changed")
    candidate, pretrain = pretrain_local_hierarchy(
        candidate,
        roles,
        output_root / "local_hierarchy_pretrain",
        seed=MODEL_SEED,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
    )
    candidate, finetune = train_gap(
        candidate,
        roles,
        output_root / "local_hierarchy_finetune",
        epochs=FINETUNE_EPOCHS,
        seed=MODEL_SEED,
        source_commit=source_commit,
        cache_sha256=cache_sha256,
    )
    results["local_hierarchy"] = {"pretrain": pretrain, "finetune": finetune}

    controls = [results[name]["validation_gap_mae_eV"] for name in ("scratch_a", "scratch_b")]
    control_mean = float(np.mean(controls))
    control_spread = float(abs(controls[0] - controls[1]))
    candidate_mae = float(finetune["validation_gap_mae_eV"])
    required_gain = max(MIN_GAIN_EV, 2.0 * control_spread)
    gain = control_mean - candidate_mae
    nominated = candidate_mae < min(controls) and gain >= required_gain
    summary = {
        "format": "molgap-qm9-edgestate-local-hierarchy-screen-v2",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "split_fingerprint": manifest["split_fingerprint"],
        "seed": MODEL_SEED,
        "gpu": torch.cuda.get_device_name(0),
        "architecture": "ogb_edge_state_structural_gps9",
        "inference_parameter_count": inference_parameters,
        "initial_model_sha256": initial_sha,
        "training_contract": {
            "scratch_epochs": SCRATCH_EPOCHS,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mask_rate": MASK_RATE,
        },
        "results": results,
        "control_mean_gap_mae_eV": control_mean,
        "control_spread_eV": control_spread,
        "candidate_gain_eV": gain,
        "required_gain_eV": required_gain,
        "pcqm_transfer_nominated": nominated,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "metrics.json", summary)
    completion = {
        **summary,
        "artifact_sha256": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and path.name != "completion_manifest.json"
        },
    }
    atomic_json(output_root / "completion_manifest.json", completion)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-cache")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--cache-sha256", required=True)
    train.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-cache":
        build_cache(args.output_root, source_commit=args.source_commit)
    elif args.command == "preflight":
        run_preflight(args.output_root, source_commit=args.source_commit)
    else:
        run_screen(
            args.cache_root,
            args.output_root,
            source_commit=args.source_commit,
            cache_sha256=args.cache_sha256,
            preflight_path=args.preflight,
        )


if __name__ == "__main__":
    main()
