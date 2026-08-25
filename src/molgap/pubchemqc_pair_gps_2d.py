"""Durable pure-2D Pair-GPS cache construction and B3LYP training.

This module deliberately keeps the architecture boundary narrow.  The input
to :class:`PairGPS2DWrapper` is a bond-topology graph only; the accepted
repaired-2M SchNet cache is used solely as an immutable source of aligned
labels and source indices while the 2D cache is being built.  No coordinates,
charges, old predictions, residual targets, or fusion heads enter this path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .pair_gps_2d import PairGPS2DWrapper


TARGET_NAMES = ("homo", "lumo", "gap")
ATOM_LIST = (6, 7, 8, 9, 16, 17, 15, 35, 14, 5, 34, 32, 33, 12, 2)
EDGE_DIM = 4
NODE_DIM = len(ATOM_LIST) + 3
SHARD_RE = re.compile(r"^graphs_(\d{7})_(\d{7})\.pt$")
FORMAT = "molgap-pubchemqc-pair-gps-2d-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _shard_paths(graph_dir: Path) -> list[Path]:
    paths = sorted(graph_dir.glob("graphs_*.pt"))
    if len(paths) != 100:
        raise RuntimeError(f"expected 100 graph shards in {graph_dir}, got {len(paths)}")
    for path in paths:
        if SHARD_RE.match(path.name) is None:
            raise RuntimeError(f"unexpected graph shard name: {path.name}")
    return paths


def _shard_range(path: Path) -> tuple[int, int]:
    match = SHARD_RE.match(path.name)
    if match is None:
        raise ValueError(f"invalid shard name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def _bond_type_map() -> dict:
    from rdkit import Chem

    return {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }


def _build_2d_graph(
    item: tuple[int, str, tuple[float, float, float]]
) -> tuple[int, dict[str, object] | None, str | None]:
    """Build one serializable heavy-atom graph record in a spawn worker."""
    source_idx, smiles, target = item
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            return source_idx, None, "parse_failed"

        atom_features = []
        for atom in molecule.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            atom_features.append(
                [float(atomic_number == value) for value in ATOM_LIST]
                + [
                    atom.GetDegree() / 4.0,
                    atom.GetFormalCharge() / 2.0,
                    float(atom.GetIsAromatic()),
                ]
            )

        bond_types = _bond_type_map()
        rows: list[int] = []
        columns: list[int] = []
        edge_features: list[list[float]] = []
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond_types.get(bond.GetBondType(), 0)
            feature = [float(bond_type == index) for index in range(EDGE_DIM)]
            rows.extend((left, right))
            columns.extend((right, left))
            edge_features.extend((feature, feature))

        # Do not return torch tensors from a multiprocessing worker.  The
        # remote IMS build uses a spawn pool whose tensor resource-sharer can
        # exhaust file descriptors on a 20K graph shard.  Plain Python lists
        # are small enough here and are assembled into PyG in the parent.
        record = {
            "x": atom_features,
            "edge_index": [rows, columns],
            "edge_attr": edge_features,
            "y": list(target),
            "source_idx": int(source_idx),
        }
        return source_idx, record, None
    except Exception as error:  # keep one bad molecule from losing a shard
        return source_idx, None, f"{type(error).__name__}:{error}"


def _build_parallel(
    work: list[tuple[int, str, tuple[float, float, float]]],
    *,
    workers: int,
    progress_label: str,
) -> Iterable[tuple[int, dict[str, object] | None, str | None]]:
    if workers <= 1:
        iterator = map(_build_2d_graph, work)
        for completed, result in enumerate(iterator, start=1):
            if completed % 1000 == 0 or completed == len(work):
                print(f"{progress_label} completed={completed}/{len(work)}", flush=True)
            yield result
        return

    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_build_2d_graph, work, chunksize=32)
        for completed, result in enumerate(iterator, start=1):
            if completed % 1000 == 0 or completed == len(work):
                print(f"{progress_label} completed={completed}/{len(work)}", flush=True)
            yield result


def _record_to_graph(record: dict[str, object]) -> Data:
    edge_features = record["edge_attr"]
    return Data(
        x=torch.tensor(record["x"], dtype=torch.float32),
        edge_index=torch.tensor(record["edge_index"], dtype=torch.long),
        edge_attr=(
            torch.tensor(edge_features, dtype=torch.float32)
            if edge_features
            else torch.zeros((0, EDGE_DIM), dtype=torch.float32)
        ),
        y=torch.tensor(record["y"], dtype=torch.float32).view(1, 3),
        source_idx=torch.tensor([int(record["source_idx"])], dtype=torch.long),
    )


def _read_manifest_smiles(manifest_path: Path, source_rows: int) -> list[str]:
    """Read only the aligned canonical SMILES column from the immutable parquet."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(manifest_path)
    if parquet.metadata.num_rows != source_rows:
        raise RuntimeError(
            f"manifest rows {parquet.metadata.num_rows} != source rows {source_rows}"
        )
    table = parquet.read(columns=["manifest_row", "canonical_smiles"])
    indices = np.asarray(table["manifest_row"].to_numpy(), dtype=np.int64)
    expected = np.arange(source_rows, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError("manifest_row is not the frozen source-index order")
    smiles = table["canonical_smiles"].to_pylist()
    if any(not isinstance(value, str) or not value for value in smiles):
        raise RuntimeError("manifest contains an empty canonical_smiles value")
    return smiles


def _primary_acceptance_report(
    primary_graph_dir: Path,
    primary_acceptance: Path,
) -> tuple[dict, list[Path], dict[str, str]]:
    acceptance = json.loads(primary_acceptance.read_text(encoding="utf-8"))
    if acceptance.get("accepted") is not True or acceptance.get("immutable") is not True:
        raise RuntimeError("primary B3LYP graph cache is not accepted and immutable")
    paths = _shard_paths(primary_graph_dir)
    if int(acceptance.get("expected_shards", -1)) != len(paths):
        raise RuntimeError("primary acceptance shard count differs from cache")
    records = acceptance.get("shards")
    if not isinstance(records, list) or len(records) != len(paths):
        raise RuntimeError("primary acceptance shard ledger is incomplete")
    by_name = {Path(record["path"]).name: record for record in records}
    hashes: dict[str, str] = {}
    for path in paths:
        record = by_name.get(path.name)
        if record is None:
            raise RuntimeError(f"primary acceptance misses {path.name}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise RuntimeError(f"primary shard hash mismatch: {path.name}")
        hashes[path.name] = actual
    return acceptance, paths, hashes


def _labels_for_primary_shard(path: Path) -> dict[int, tuple[float, float, float]]:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    labels: dict[int, tuple[float, float, float]] = {}
    for graph in graphs:
        source_idx = int(graph.source_idx.view(-1)[0])
        target = graph.y.view(-1, 3).view(-1).float()
        if source_idx in labels:
            raise RuntimeError(f"duplicate primary source_idx {source_idx}")
        if not torch.isfinite(target).all():
            raise RuntimeError(f"non-finite primary target at source_idx {source_idx}")
        labels[source_idx] = tuple(float(value) for value in target.tolist())
    return labels


def build_pair_gps_2d_cache(
    *,
    manifest_path: Path,
    primary_graph_dir: Path,
    primary_acceptance: Path,
    output_dir: Path,
    source_rows: int = 2_000_000,
    workers: int = 14,
) -> dict:
    """Build a resumable pure-2D cache aligned to accepted B3LYP source rows."""
    if source_rows != 2_000_000:
        raise ValueError("the repaired-2M source contract is fixed at 2,000,000 rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graph_shards"
    reports_dir = graph_dir / "reports"
    graph_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    primary_acceptance_data, primary_paths, primary_hashes = _primary_acceptance_report(
        primary_graph_dir, primary_acceptance
    )
    smiles = _read_manifest_smiles(manifest_path, source_rows)
    input_contract = {
        "format": FORMAT,
        "source_rows": source_rows,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "primary_acceptance_path": str(primary_acceptance),
        "primary_acceptance_sha256": sha256_file(primary_acceptance),
        "primary_graph_ledger": primary_hashes,
        "atom_list": list(ATOM_LIST),
        "node_feature_dim": NODE_DIM,
        "edge_feature_dim": EDGE_DIM,
        "label_source": "accepted_primary_b3lyp_graph_y_by_source_idx",
        "geometry": "none",
    }
    contract_path = output_dir / "input_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != input_contract:
            raise RuntimeError("2D cache input contract changed on resume")
    else:
        _atomic_json(input_contract, contract_path)

    primary_graph_count = int(primary_acceptance_data["accepted_rows"])
    completion_path = output_dir / "build_completion.json"
    progress_path = output_dir / "progress.json"
    completed_reports: list[dict] = []
    for primary_path in primary_paths:
        start, stop = _shard_range(primary_path)
        graph_path = graph_dir / primary_path.name
        report_path = reports_dir / f"{primary_path.stem}.json"
        if graph_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("status") == "complete"
                and report.get("format") == FORMAT
                and report.get("primary_sha256") == primary_hashes[primary_path.name]
                and report.get("sha256") == sha256_file(graph_path)
            ):
                completed_reports.append(report)
                print(f"reuse completed 2D shard {graph_path.name}", flush=True)
                continue

        labels = _labels_for_primary_shard(primary_path)
        work = [
            (source_idx, smiles[source_idx], target)
            for source_idx, target in labels.items()
        ]
        built: list[Data] = []
        failures: list[dict[str, object]] = []
        for source_idx, graph, error in _build_parallel(
            work, workers=max(1, workers), progress_label=f"2D {start}:{stop}"
        ):
            if graph is None:
                failures.append({"source_idx": int(source_idx), "error": error})
            else:
                built.append(_record_to_graph(graph))
        built.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
        if len({int(graph.source_idx.item()) for graph in built}) != len(built):
            raise RuntimeError(f"duplicate 2D source_idx in {primary_path.name}")
        _atomic_torch_save(built, graph_path)
        report = {
            "status": "complete",
            "format": FORMAT,
            "start": start,
            "stop": stop,
            "primary_sha256": primary_hashes[primary_path.name],
            "requested_primary_graphs": len(labels),
            "graphs": len(built),
            "failed": len(failures),
            "failure_source_idx": failures,
            "path": graph_path.name,
            "bytes": graph_path.stat().st_size,
            "sha256": sha256_file(graph_path),
        }
        _atomic_json(report, report_path)
        completed_reports.append(report)
        _atomic_json(
            {
                "status": "running",
                "format": FORMAT,
                "completed_shards": len(completed_reports),
                "total_shards": len(primary_paths),
                "primary_graphs": sum(item["requested_primary_graphs"] for item in completed_reports),
                "graphs": sum(item["graphs"] for item in completed_reports),
                "failed": sum(item["failed"] for item in completed_reports),
            },
            progress_path,
        )
        print(
            f"2D shard {primary_path.name}: graphs={len(built)} "
            f"failed={len(failures)}",
            flush=True,
        )

    completed_reports.sort(key=lambda item: int(item["start"]))
    total_graphs = sum(int(item["graphs"]) for item in completed_reports)
    total_failed = sum(int(item["failed"]) for item in completed_reports)
    if len(completed_reports) != len(primary_paths):
        raise RuntimeError("not all 2D shards completed")
    if total_graphs + total_failed != primary_graph_count:
        raise RuntimeError("2D graph counts do not reconcile to accepted primary rows")
    completion = {
        "status": "complete",
        "format": FORMAT,
        "source_rows": source_rows,
        "primary_accepted_rows": primary_graph_count,
        "graphs": total_graphs,
        "failed": total_failed,
        "shards": len(completed_reports),
        "input_contract_sha256": sha256_file(contract_path),
        "reports": completed_reports,
    }
    _atomic_json(completion, completion_path)
    _atomic_json(completion, progress_path)
    return completion


def _source_roles(source_rows: int, split_seed: int) -> np.ndarray:
    permutation = np.random.default_rng(split_seed).permutation(source_rows)
    train = int(0.8 * source_rows)
    validation = int(0.1 * source_rows)
    roles = np.full(source_rows, 2, dtype=np.int8)
    roles[permutation[:train]] = 0
    roles[permutation[train : train + validation]] = 1
    return roles


def _metric_vector(absolute_error: torch.Tensor, count: int) -> dict[str, float]:
    values = (absolute_error / max(count, 1)).tolist()
    result = {name: float(values[index]) for index, name in enumerate(TARGET_NAMES)}
    result["average"] = float(sum(values) / len(values))
    return result


def _load_accepted_2d_cache(cache_dir: Path) -> tuple[dict, list[Path]]:
    acceptance_path = cache_dir / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("accepted") is not True or acceptance.get("geometry") != "none":
        raise RuntimeError("2D graph cache is not accepted as pure topology")
    paths = _shard_paths(cache_dir / "graph_shards")
    records = {Path(item["path"]).name: item for item in acceptance.get("shards", [])}
    if len(records) != len(paths):
        raise RuntimeError("2D acceptance ledger is incomplete")
    for path in paths:
        record = records.get(path.name)
        if record is None or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"2D accepted shard hash mismatch: {path.name}")
    return acceptance, paths


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_config() -> dict[str, object]:
    return {
        "in_channels": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "hidden_channels": 256,
        "pair_channels": 96,
        "num_layers": 10,
        "num_heads": 8,
        "dropout": 0.05,
        "n_targets": 3,
        "pooling": "mean",
        "path_steps": 5,
        "triplet_rank": 16,
    }


def _new_model(device: torch.device) -> PairGPS2DWrapper:
    return PairGPS2DWrapper(**_model_config()).to(device)


def _size_bucket_batches(
    graphs: list[Data], batch_size: int, *, seed: int
) -> list[list[int]]:
    """Group similar node counts while randomizing batch and sample order."""
    indices = sorted(range(len(graphs)), key=lambda index: int(graphs[index].num_nodes))
    batches = [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
    rng = random.Random(seed)
    for batch in batches:
        rng.shuffle(batch)
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def _evaluate_role(
    model: nn.Module,
    paths: list[Path],
    roles: np.ndarray,
    role: int,
    device: torch.device,
    batch_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    collect_payload: bool = False,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    model.eval()
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    absolute_error = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    source_indices: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for shard_index, path in enumerate(paths):
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        selected = [
            graph
            for graph in graphs
            if int(roles[int(graph.source_idx.view(-1)[0])]) == role
        ]
        if not selected:
            continue
        loader = DataLoader(
            selected,
            batch_sampler=_size_bucket_batches(
                selected, batch_size, seed=role * 1_000_003 + shard_index
            ),
            num_workers=0,
            pin_memory=True,
        )
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.view(-1, 3)
            prediction = prediction.float() * target_std_device + target_mean_device
            target = target.float()
            absolute_error += prediction.sub(target).abs().sum(dim=0).double()
            count += int(batch.num_graphs)
            if collect_payload:
                source_indices.append(batch.source_idx.view(-1).long().cpu())
                predictions.append(prediction.cpu())
                targets.append(target.cpu())
    if count == 0:
        raise RuntimeError(f"empty role {role} in 2D evaluation")
    if not bool(torch.isfinite(absolute_error).all().item()):
        raise RuntimeError("Pair-GPS 2D evaluation produced non-finite output")
    empty_indices = torch.empty(0, dtype=torch.long)
    empty_payload = torch.empty((0, 6), dtype=torch.float32)
    return (
        absolute_error.cpu(),
        count,
        torch.cat(source_indices) if source_indices else empty_indices,
        (
            torch.cat((torch.cat(predictions), torch.cat(targets)), dim=1)
            if predictions
            else empty_payload
        ),
    )


def train_pair_gps_2d(
    *,
    cache_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    patience: int = 6,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    checkpoint_every_shards: int = 10,
    learning_rate: float = 4e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    split_seed: int = 42,
    resume: bool = True,
) -> dict:
    """Train the direct pure-2D architecture from scratch on accepted B3LYP rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PubChemQC Pair-GPS 2D training")
    acceptance, paths = _load_accepted_2d_cache(cache_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eval_batch_size is None:
        eval_batch_size = batch_size
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if checkpoint_every_shards < 1:
        raise ValueError("checkpoint_every_shards must be positive")
    source_rows = int(acceptance["source_rows"])
    roles = _source_roles(source_rows, split_seed)
    target_mean = torch.tensor(acceptance["target_mean"], dtype=torch.float32)
    target_std = torch.tensor(acceptance["target_std"], dtype=torch.float32).clamp_min(1e-6)
    config = {
        "trainer_revision": 2,
        "format": FORMAT,
        "cache_acceptance_sha256": sha256_file(cache_dir / "acceptance.json"),
        "model_config": _model_config(),
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "checkpoint_every_shards": checkpoint_every_shards,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "split_seed": split_seed,
        "source_rows": source_rows,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "amp": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.pt"
    _set_seed(seed)
    device = torch.device("cuda")
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    model = _new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    criterion = nn.L1Loss()
    best_state = None
    best_epoch = -1
    best_mae = float("inf")
    wait = 0
    log: list[dict[str, object]] = []
    start_epoch = 0
    resume_phase = "new"
    resume_next_shard = 0
    resume_total_loss = 0.0
    resume_train_count = 0
    resume_global_step = 0
    resume_epoch_elapsed = 0.0

    def save_checkpoint(
        *,
        phase: str,
        epoch: int,
        next_shard: int,
        total_loss: float,
        train_count: int,
        global_step: int,
        epoch_elapsed_s: float,
    ) -> None:
        _atomic_torch_save(
            {
                "phase": phase,
                "epoch": epoch,
                "next_shard": next_shard,
                "total_loss": total_loss,
                "train_count": train_count,
                "global_step": global_step,
                "epoch_elapsed_s": epoch_elapsed_s,
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_state": best_state,
                "best_mae": best_mae,
                "best_epoch": best_epoch,
                "wait": wait,
                "log": log,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_path,
        )

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config:
            raise RuntimeError("Pair-GPS 2D resume contract changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_mae"])
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        resume_phase = str(checkpoint.get("phase", "epoch_complete"))
        checkpoint_epoch = int(checkpoint["epoch"])
        start_epoch = checkpoint_epoch + 1 if resume_phase == "epoch_complete" else checkpoint_epoch
        resume_next_shard = int(checkpoint.get("next_shard", 0))
        resume_total_loss = float(checkpoint.get("total_loss", 0.0))
        resume_train_count = int(checkpoint.get("train_count", 0))
        resume_global_step = int(checkpoint.get("global_step", 0))
        resume_epoch_elapsed = float(checkpoint.get("epoch_elapsed_s", 0.0))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )
        print(
            f"resume pair_gps_2d phase={resume_phase} epoch={start_epoch} "
            f"next_shard={resume_next_shard} step={resume_global_step}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        continuing_epoch = epoch == start_epoch and resume_phase in {"train", "validation"}
        total_loss = resume_total_loss if continuing_epoch else 0.0
        train_count = resume_train_count if continuing_epoch else 0
        global_step = resume_global_step if continuing_epoch else 0
        epoch_elapsed_before = resume_epoch_elapsed if continuing_epoch else 0.0

        if not (continuing_epoch and resume_phase == "validation"):
            first_shard = resume_next_shard if continuing_epoch else 0
            model.train()
            for shard_index in range(first_shard, len(paths)):
                path = paths[shard_index]
                graphs = torch.load(path, map_location="cpu", weights_only=False)
                selected = [
                    graph
                    for graph in graphs
                    if int(roles[int(graph.source_idx.view(-1)[0])]) == 0
                ]
                if not selected:
                    continue
                loader = DataLoader(
                    selected,
                    batch_sampler=_size_bucket_batches(
                        selected,
                        batch_size,
                        seed=seed + epoch * 100_003 + shard_index,
                    ),
                    num_workers=0,
                    pin_memory=True,
                )
                shard_loss = torch.zeros((), dtype=torch.float64, device=device)
                shard_count = 0
                for batch in loader:
                    batch = batch.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    target = (
                        batch.y.view(-1, 3) - target_mean_device
                    ) / target_std_device
                    prediction = model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                    loss = criterion(prediction, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    shard_loss += loss.detach().double() * int(batch.num_graphs)
                    shard_count += int(batch.num_graphs)
                    global_step += 1

                shard_loss_value = float(shard_loss.item())
                if not np.isfinite(shard_loss_value):
                    raise RuntimeError(
                        f"non-finite Pair-GPS 2D loss at epoch {epoch} "
                        f"shard {shard_index}"
                    )
                total_loss += shard_loss_value
                train_count += shard_count
                elapsed_s = epoch_elapsed_before + time.perf_counter() - started
                print(
                    f"pair_gps_2d ep{epoch:02d} shard={shard_index + 1:03d}/"
                    f"{len(paths):03d} rows={train_count} step={global_step} "
                    f"elapsed={elapsed_s:.1f}s",
                    flush=True,
                )
                if (
                    (shard_index + 1) % checkpoint_every_shards == 0
                    or shard_index + 1 == len(paths)
                ):
                    save_checkpoint(
                        phase="train",
                        epoch=epoch,
                        next_shard=shard_index + 1,
                        total_loss=total_loss,
                        train_count=train_count,
                        global_step=global_step,
                        epoch_elapsed_s=elapsed_s,
                    )

            expected_train_count = int(acceptance["split_accepted_rows"]["train"])
            if train_count != expected_train_count:
                raise RuntimeError(
                    f"Pair-GPS 2D train rows changed: {train_count} != "
                    f"{expected_train_count}"
                )
            scheduler.step()
            save_checkpoint(
                phase="validation",
                epoch=epoch,
                next_shard=len(paths),
                total_loss=total_loss,
                train_count=train_count,
                global_step=global_step,
                epoch_elapsed_s=epoch_elapsed_before + time.perf_counter() - started,
            )

        validation_error, validation_count, _, _ = _evaluate_role(
            model,
            paths,
            roles,
            1,
            device,
            eval_batch_size,
            target_mean,
            target_std,
        )
        validation_metrics = _metric_vector(validation_error, validation_count)
        val_mae = validation_metrics["average"]
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
            "train_normalized_l1": total_loss / max(train_count, 1),
            "train_rows": train_count,
            "validation_rows": validation_count,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": epoch_elapsed_before + time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        save_checkpoint(
            phase="epoch_complete",
            epoch=epoch,
            next_shard=0,
            total_loss=0.0,
            train_count=0,
            global_step=0,
            epoch_elapsed_s=0.0,
        )
        print(
            f"pair_gps_2d ep{epoch:02d} train={row['train_normalized_l1']:.6f} "
            f"val={val_mae:.6f}eV elapsed={row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break
        resume_phase = "new"
        resume_next_shard = 0
        resume_total_loss = 0.0
        resume_train_count = 0
        resume_global_step = 0
        resume_epoch_elapsed = 0.0

    if best_state is None:
        raise RuntimeError("Pair-GPS 2D training produced no best state")
    model.load_state_dict(best_state, strict=True)
    test_error, test_count, test_source_idx, test_payload = _evaluate_role(
        model,
        paths,
        roles,
        2,
        device,
        eval_batch_size,
        target_mean,
        target_std,
        collect_payload=True,
    )
    validation_error, validation_count, _, _ = _evaluate_role(
        model,
        paths,
        roles,
        1,
        device,
        eval_batch_size,
        target_mean,
        target_std,
    )
    test_metrics = _metric_vector(test_error, test_count)
    validation_metrics = _metric_vector(validation_error, validation_count)
    _atomic_torch_save(
        {
            "source_idx": test_source_idx,
            "prediction": test_payload[:, :3],
            "target": test_payload[:, 3:],
        },
        predictions_path,
    )
    _atomic_torch_save(best_state, model_path)
    result = {
        "experiment": "pubchemqc_pair_gps_2d",
        "architecture": "PairGPS2D",
        "geometry": "pure_2d_bond_topology",
        "target": "PubChemQC B3LYP/6-31G* HOMO/LUMO/Gap",
        "seed": seed,
        "split_seed": split_seed,
        "source_rows": source_rows,
        "accepted_graph_rows": int(acceptance["accepted_rows"]),
        "split_rows": {
            "train": int(np.sum(roles == 0)),
            "validation": int(np.sum(roles == 1)),
            "test": int(np.sum(roles == 2)),
            "train_accepted": int(acceptance["split_accepted_rows"]["train"]),
            "validation_accepted": validation_count,
            "test_accepted": test_count,
        },
        "target_names": list(TARGET_NAMES),
        "target_units": "eV",
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "model_config": _model_config(),
        "training_config": config,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "validation": validation_metrics,
        "test": test_metrics,
        "log": log,
        "artifacts": {
            "cache_acceptance": str(cache_dir / "acceptance.json"),
            "checkpoint": str(checkpoint_path),
            "model": str(model_path),
            "test_predictions": str(predictions_path),
        },
    }
    _atomic_json(result, metrics_path)
    return result


def preflight_pair_gps_2d(
    *,
    cache_dir: Path,
    output_path: Path,
    batch_size: int = 4,
) -> dict:
    """Benchmark finite size-bucketed steps on representative real graphs."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Pair-GPS 2D preflight")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    acceptance, paths = _load_accepted_2d_cache(cache_dir)
    candidates = []
    sample_per_shard = max(batch_size * 8, 256)
    for path in (paths[0], paths[len(paths) // 2], paths[-1]):
        candidates.extend(
            torch.load(path, map_location="cpu", weights_only=False)[:sample_per_shard]
        )
    candidates.sort(key=lambda graph: int(graph.num_nodes))
    device = torch.device("cuda")
    target_mean = torch.tensor(acceptance["target_mean"], dtype=torch.float32, device=device)
    target_std = torch.tensor(
        acceptance["target_std"], dtype=torch.float32, device=device
    ).clamp_min(1e-6)
    batch_sizes = sorted(
        {
            max(4, batch_size // 4),
            max(4, batch_size // 2),
            batch_size,
        }
    )
    benchmarks = []
    n_params = 0
    for candidate_batch_size in batch_sizes:
        grouped = [
            candidates[start : start + candidate_batch_size]
            for start in range(0, len(candidates), candidate_batch_size)
            if len(candidates[start : start + candidate_batch_size])
            == candidate_batch_size
        ]
        if not grouped:
            raise RuntimeError(
                f"no full preflight batch for size {candidate_batch_size}"
            )
        positions = sorted(
            {
                int((len(grouped) - 1) * fraction)
                for fraction in (0.25, 0.50, 0.75, 1.00)
            }
        )
        _set_seed(42 + candidate_batch_size)
        model = _new_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=1e-5
        )
        n_params = sum(parameter.numel() for parameter in model.parameters())
        rows = []
        for position in positions:
            graphs = grouped[position]
            loader = DataLoader(
                graphs,
                batch_size=candidate_batch_size,
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader)).to(device, non_blocking=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            target = (batch.y.view(-1, 3) - target_mean) / target_std
            loss = nn.functional.l1_loss(prediction, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite = (
                torch.isfinite(prediction).all()
                & torch.isfinite(loss)
                & torch.isfinite(grad_norm)
            )
            if not bool(finite.item()):
                raise RuntimeError("Pair-GPS 2D preflight produced a non-finite tensor")
            optimizer.step()
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - started
            rows.append(
                {
                    "quantile_batch": position,
                    "total_nodes": int(batch.num_nodes),
                    "max_nodes": max(int(graph.num_nodes) for graph in graphs),
                    "elapsed_s": elapsed_s,
                    "graphs_per_s": int(batch.num_graphs) / max(elapsed_s, 1e-9),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )
        elapsed_values = sorted(float(row["elapsed_s"]) for row in rows)
        benchmarks.append(
            {
                "batch_size": candidate_batch_size,
                "steps": rows,
                "median_step_elapsed_s": elapsed_values[len(elapsed_values) // 2],
                "aggregate_graphs_per_s": (
                    candidate_batch_size * len(rows)
                    / max(sum(float(row["elapsed_s"]) for row in rows), 1e-9)
                ),
                "peak_cuda_bytes": max(int(row["peak_cuda_bytes"]) for row in rows),
            }
        )
        del model, optimizer
        torch.cuda.empty_cache()
    recommended = max(
        benchmarks, key=lambda item: float(item["aggregate_graphs_per_s"])
    )
    report = {
        "status": "complete",
        "architecture": "PairGPS2D",
        "geometry": "none",
        "cache_acceptance_sha256": sha256_file(cache_dir / "acceptance.json"),
        "sample_candidates": len(candidates),
        "tested_batch_sizes": batch_sizes,
        "batch_benchmarks": benchmarks,
        "recommended_batch_size": int(recommended["batch_size"]),
        "model_config": _model_config(),
        "n_params": n_params,
        "device": torch.cuda.get_device_name(0),
        "finite": True,
        "target": "PubChemQC B3LYP/6-31G* HOMO/LUMO/Gap",
        "input_acceptance": {
            "format": acceptance["format"],
            "accepted_rows": int(acceptance["accepted_rows"]),
            "split_accepted_rows": acceptance["split_accepted_rows"],
            "geometry": acceptance["geometry"],
        },
    }
    _atomic_json(report, output_path)
    return report


def accept_pair_gps_2d_cache(
    *,
    cache_dir: Path,
    source_rows: int = 2_000_000,
    split_seed: int = 42,
) -> dict:
    """Independently validate pure-2D structure, hashes, labels, and split stats."""
    completion = json.loads((cache_dir / "build_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("format") != FORMAT:
        raise RuntimeError("2D cache build is not complete")
    paths = _shard_paths(cache_dir / "graph_shards")
    reports = {Path(item["path"]).name: item for item in completion.get("reports", [])}
    if len(reports) != len(paths):
        raise RuntimeError("2D completion ledger is incomplete")
    roles = _source_roles(source_rows, split_seed)
    seen: set[int] = set()
    counts = [0, 0, 0]
    train_target_sum = np.zeros(3, dtype=np.float64)
    train_target_sq_sum = np.zeros(3, dtype=np.float64)
    train_target_count = 0
    shard_records = []
    for path in paths:
        report = reports.get(path.name)
        if report is None or sha256_file(path) != report.get("sha256"):
            raise RuntimeError(f"2D shard hash mismatch: {path.name}")
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        local_seen: set[int] = set()
        for graph in graphs:
            keys = set(graph.keys())
            if not {"x", "edge_index", "edge_attr", "y", "source_idx"} <= keys:
                raise RuntimeError(f"missing pure-2D fields in {path.name}")
            if {"pos", "z", "charges"} & keys:
                raise RuntimeError(f"3D field found in pure-2D graph {path.name}")
            source_idx = int(graph.source_idx.view(-1)[0])
            if source_idx in seen or source_idx in local_seen:
                raise RuntimeError(f"duplicate 2D source_idx {source_idx}")
            if source_idx < 0 or source_idx >= source_rows:
                raise RuntimeError(f"source_idx outside source rows: {source_idx}")
            if graph.x.ndim != 2 or graph.x.shape[1] != NODE_DIM:
                raise RuntimeError(f"unexpected x shape in {path.name}")
            if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
                raise RuntimeError(f"unexpected edge_index shape in {path.name}")
            if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != EDGE_DIM:
                raise RuntimeError(f"unexpected edge_attr shape in {path.name}")
            target = graph.y.view(-1, 3).view(-1).double()
            if target.numel() != 3 or not torch.isfinite(target).all():
                raise RuntimeError(f"invalid target in {path.name}")
            role = int(roles[source_idx])
            counts[role] += 1
            values = target.numpy()
            if role == 0:
                train_target_sum += values
                train_target_sq_sum += values * values
                train_target_count += 1
            seen.add(source_idx)
            local_seen.add(source_idx)
        shard_records.append(
            {
                "path": path.name,
                "rows": len(graphs),
                "sha256": report["sha256"],
                "primary_sha256": report["primary_sha256"],
            }
        )
    if len(seen) != int(completion["graphs"]):
        raise RuntimeError("2D graph count does not match source-index coverage")
    if train_target_count < 2:
        raise RuntimeError("not enough accepted training targets for normalization")
    n = float(train_target_count)
    mean = train_target_sum / n
    variance = (
        train_target_sq_sum - train_target_sum * train_target_sum / n
    ) / (n - 1.0)
    std = np.sqrt(np.maximum(variance, 1e-12))
    ledger = hashlib.sha256()
    for record in shard_records:
        ledger.update(
            f"{record['path']}\0{record['rows']}\0{record['sha256']}\n".encode("ascii")
        )
    result = {
        "accepted": True,
        "format": FORMAT,
        "geometry": "none",
        "source_rows": source_rows,
        "accepted_rows": len(seen),
        "expected_accepted_rows": int(completion["graphs"]),
        "failed_2d_rows": int(completion["failed"]),
        "expected_shards": len(paths),
        "node_feature_dim": NODE_DIM,
        "edge_feature_dim": EDGE_DIM,
        "atom_list": list(ATOM_LIST),
        "target_names": list(TARGET_NAMES),
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "split_seed": split_seed,
        "split_source_rows": {
            "train": int(np.sum(roles == 0)),
            "validation": int(np.sum(roles == 1)),
            "test": int(np.sum(roles == 2)),
        },
        "split_accepted_rows": {
            "train": counts[0],
            "validation": counts[1],
            "test": counts[2],
        },
        "input_contract_sha256": sha256_file(cache_dir / "input_contract.json"),
        "cache_ledger_sha256": ledger.hexdigest(),
        "normalization_rows": train_target_count,
        "shards": shard_records,
    }
    _atomic_json(result, cache_dir / "acceptance.json")
    return result
