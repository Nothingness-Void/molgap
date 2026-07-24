"""Gap-only specialization on frozen molecular embeddings."""
from __future__ import annotations

import hashlib
import json
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .distillation import atomic_json_write, atomic_torch_save, sha256_file
from .gps import GPSWrapper
from .graphs import smiles_to_2d_pyg
from .router_sampling import compute_scaffold_keys


def _stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_gap_graph_cache(
    table: pd.DataFrame,
    *,
    smiles_column: str,
    gap_column: str,
    out_path: Path,
    progress_path: Path,
    shard_dir: Path,
    shard_size: int = 10_000,
) -> dict:
    """Build a resumable 2D cache whose first two targets are intentionally masked."""
    required = {smiles_column, gap_column}
    if not required <= set(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    start = int(progress_path.read_text().strip()) if progress_path.is_file() else 0
    valid_rows = 0
    for shard_start in range(start, len(table), shard_size):
        shard_end = min(shard_start + shard_size, len(table))
        shard_path = shard_dir / f"part-{shard_start:07d}-{shard_end:07d}.pt"
        if not shard_path.is_file():
            graphs = []
            for source_idx, row in table.iloc[shard_start:shard_end].iterrows():
                graph = smiles_to_2d_pyg(str(row[smiles_column]))
                if graph is None:
                    continue
                graph.y = torch.tensor(
                    [[float("nan"), float("nan"), float(row[gap_column])]],
                    dtype=torch.float32,
                )
                graph.target_mask = torch.tensor([[False, False, True]])
                graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
                graphs.append(graph)
            atomic_torch_save(graphs, shard_path)
        else:
            graphs = torch.load(shard_path, map_location="cpu", weights_only=False)
        valid_rows += len(graphs)
        temporary = progress_path.with_name(f".{progress_path.name}.tmp")
        temporary.write_text(str(shard_end), encoding="utf-8")
        os.replace(temporary, progress_path)
        print(f"gap graphs {shard_end:,}/{len(table):,}; valid={valid_rows:,}", flush=True)

    merged = []
    for shard_path in sorted(shard_dir.glob("part-*.pt")):
        merged.extend(torch.load(shard_path, map_location="cpu", weights_only=False))
    expected_source = torch.arange(len(table), dtype=torch.long)
    actual_source = torch.as_tensor(
        [int(graph.source_idx.view(-1)[0]) for graph in merged], dtype=torch.long
    )
    if not torch.equal(actual_source, expected_source):
        raise ValueError("Gap graph cache must contain every row in source order")
    atomic_torch_save(merged, out_path)
    report = {
        "complete": True,
        "rows": len(merged),
        "source_idx_min": int(actual_source.min()),
        "source_idx_max": int(actual_source.max()),
        "gap_column": gap_column,
        "graph_cache": str(out_path),
        "graph_cache_sha256": sha256_file(out_path),
    }
    atomic_json_write(report, out_path.with_suffix(".json"))
    return report


def load_embedding_parts(manifest_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete embedding manifest: {manifest_path}")
    embeddings, source_indices = [], []
    for entry in manifest["parts"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path.name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        embeddings.append(payload["embeddings"])
        source_indices.append(payload["source_idx"].long())
    embedding = torch.cat(embeddings)
    source_idx = torch.cat(source_indices)
    if len(embedding) != int(manifest["rows"]) or len(torch.unique(source_idx)) != len(source_idx):
        raise ValueError(f"Invalid embedding rows in {manifest_path}")
    return embedding, source_idx


def scaffold_split(
    smiles: Sequence[str],
    *,
    seed: int,
    workers: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Create a deterministic scaffold-disjoint split with approximate row quotas."""
    keys = compute_scaffold_keys(list(smiles), workers=workers)
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(str(key), []).append(index)
    ordered = sorted(groups, key=lambda key: _stable_hash(key, seed))
    train_target = int(train_fraction * len(keys))
    validation_target = int(validation_fraction * len(keys))
    result = {"train": [], "validation": [], "test": []}
    for key in ordered:
        destination = (
            "train"
            if len(result["train"]) < train_target
            else "validation"
            if len(result["validation"]) < validation_target
            else "test"
        )
        result[destination].extend(groups[key])
    arrays = {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in result.items()
    }
    scaffold_sets = {name: set(keys[index] for index in values) for name, values in arrays.items()}
    if any(
        scaffold_sets[left] & scaffold_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("Scaffold split leakage detected")
    return arrays


def _mae(head: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    absolute = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            prediction = head(x[start:start + batch_size].float().to(device)).float().cpu()
            absolute.append((prediction - y[start:start + batch_size]).abs())
    return torch.cat(absolute).mean(dim=0).numpy()


def train_gap_specialist_head(
    *,
    base_model_path: Path,
    b3_embedding_manifest: Path,
    b3_graph_path: Path,
    pcqm_embedding_manifest: Path,
    pcqm_table_path: Path,
    run_dir: Path,
    model_out: Path,
    b3_train_rows: int = 200_000,
    b3_validation_rows: int = 20_000,
    pcqm_weight: float = 0.30,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    scaffold_workers: int = 8,
) -> dict:
    """Adapt only the GPS output head using B3LYP rehearsal and masked PCQM Gap."""
    if not 0.0 < pcqm_weight < 1.0:
        raise ValueError("pcqm_weight must be between zero and one")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_state = torch.load(base_model_path, map_location="cpu", weights_only=True)
    model = GPSWrapper(hidden_channels=192, num_layers=7, num_heads=4, dropout=0.05)
    model.load_state_dict(base_state)
    head = model.head.to(device)

    b3_x, b3_source = load_embedding_parts(b3_embedding_manifest)
    if not torch.equal(b3_source, torch.arange(len(b3_source))):
        raise ValueError("B3LYP embeddings are not contiguous")
    graphs = torch.load(b3_graph_path, map_location="cpu", weights_only=False)
    if len(graphs) < len(b3_x):
        raise ValueError("B3LYP graph cache is shorter than the embedding prefix")
    graphs = graphs[:len(b3_x)]
    graph_source = torch.as_tensor(
        [int(graph.source_idx.view(-1)[0]) for graph in graphs],
        dtype=torch.long,
    )
    if not torch.equal(graph_source, b3_source):
        raise ValueError("B3LYP graph and embedding source indices differ")
    b3_y = torch.cat([graph.y.float().cpu() for graph in graphs])
    del graphs

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(b3_x))
    b3_train_idx = permutation[:b3_train_rows]
    b3_val_idx = permutation[b3_train_rows:b3_train_rows + b3_validation_rows]
    b3_train_x, b3_train_y = b3_x[b3_train_idx], b3_y[b3_train_idx]
    b3_val_x, b3_val_y = b3_x[b3_val_idx], b3_y[b3_val_idx]
    del b3_x, b3_y

    pcqm = pd.read_parquet(pcqm_table_path)
    pcqm_x, pcqm_source = load_embedding_parts(pcqm_embedding_manifest)
    if not torch.equal(pcqm_source, torch.arange(len(pcqm))):
        raise ValueError("PCQM table and embeddings are not aligned")
    split = scaffold_split(
        pcqm.canonical_smiles.astype(str).tolist(),
        seed=seed,
        workers=scaffold_workers,
    )
    pcqm_y = torch.as_tensor(pcqm.homolumogap.to_numpy(np.float32)[:, None])
    pcqm_train_x, pcqm_train_y = pcqm_x[split["train"]], pcqm_y[split["train"]]
    pcqm_val_x, pcqm_val_y = pcqm_x[split["validation"]], pcqm_y[split["validation"]]

    b3_loader = DataLoader(
        TensorDataset(b3_train_x, b3_train_y), batch_size=batch_size, shuffle=True
    )
    pcqm_loader = DataLoader(
        TensorDataset(pcqm_train_x, pcqm_train_y), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    checkpoint_path = run_dir / "training_state.pt"
    metrics_path = run_dir / "metrics.json"
    config = {
        "base_model_sha256": sha256_file(base_model_path),
        "b3_embedding_manifest_sha256": sha256_file(b3_embedding_manifest),
        "pcqm_embedding_manifest_sha256": sha256_file(pcqm_embedding_manifest),
        "pcqm_table_sha256": sha256_file(pcqm_table_path),
        "b3_train_rows": b3_train_rows,
        "b3_validation_rows": b3_validation_rows,
        "pcqm_weight": pcqm_weight,
        "seed": seed,
    }
    start_epoch, best_score, best_epoch, wait, best_head, log = 0, float("inf"), -1, 0, None, []
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["config"] != config:
            raise ValueError("Resume checkpoint configuration differs")
        head.load_state_dict(state["head_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["next_epoch"])
        best_score, best_epoch, wait = float(state["best_score"]), int(state["best_epoch"]), int(state["wait"])
        best_head, log = state["best_head"], list(state["log"])

    for epoch in range(start_epoch, epochs):
        started = time.time()
        head.train()
        pcqm_batches = cycle(pcqm_loader)
        total = 0.0
        for b3_batch_x, b3_batch_y in b3_loader:
            pcqm_batch_x, pcqm_batch_y = next(pcqm_batches)
            optimizer.zero_grad(set_to_none=True)
            b3_prediction = head(b3_batch_x.float().to(device))
            pcqm_prediction = head(pcqm_batch_x.float().to(device))[:, 2:3]
            loss = (
                (1.0 - pcqm_weight) * criterion(b3_prediction, b3_batch_y.to(device))
                + pcqm_weight * criterion(pcqm_prediction, pcqm_batch_y.to(device))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        b3_val_mae = _mae(head, b3_val_x, b3_val_y, batch_size, device)
        pcqm_val_gap = float(
            _mae(
                head,
                pcqm_val_x,
                torch.cat(
                    [
                        torch.full_like(pcqm_val_y, float("nan")),
                        torch.full_like(pcqm_val_y, float("nan")),
                        pcqm_val_y,
                    ],
                    dim=1,
                ),
                batch_size,
                device,
            )[2]
        )
        score = (1.0 - pcqm_weight) * float(b3_val_mae.mean()) + pcqm_weight * pcqm_val_gap
        improved = score < best_score
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_objective": total / max(len(b3_loader), 1),
            "b3_validation_mae_eV": {
                "homo": float(b3_val_mae[0]),
                "lumo": float(b3_val_mae[1]),
                "gap": float(b3_val_mae[2]),
                "average": float(b3_val_mae.mean()),
            },
            "pcqm_validation_gap_mae_eV": pcqm_val_gap,
            "selection_score": score,
            "seconds": time.time() - started,
        }
        log.append(row)
        atomic_torch_save(
            {
                "config": config,
                "next_epoch": epoch + 1,
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "wait": wait,
                "best_head": best_head,
                "log": log,
            },
            checkpoint_path,
        )
        atomic_json_write(
            {"complete": False, "config": config, "best_epoch": best_epoch, "log": log},
            metrics_path,
        )
        print(
            f"ep{epoch:03d} score={score:.5f} b3={b3_val_mae.mean():.5f} "
            f"pcqm_gap={pcqm_val_gap:.5f} best={best_score:.5f}@{best_epoch}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= patience:
            break

    if best_head is None:
        raise RuntimeError("No finite specialist head was produced")
    head.load_state_dict(best_head)
    candidate_state = dict(base_state)
    for key, value in head.state_dict().items():
        candidate_state[f"head.{key}"] = value
    atomic_torch_save(candidate_state, model_out)
    result = {
        "complete": True,
        "config": config,
        "split": {
            "b3_train": len(b3_train_idx),
            "b3_validation": len(b3_val_idx),
            "pcqm_train": len(split["train"]),
            "pcqm_validation": len(split["validation"]),
            "pcqm_test_locked": len(split["test"]),
        },
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "model": {"path": str(model_out), "sha256": sha256_file(model_out)},
        "log": log,
    }
    atomic_json_write(result, metrics_path)
    atomic_json_write(
        {
            "complete": True,
            "model": result["model"],
            "metrics": str(metrics_path),
            "checkpoint": str(checkpoint_path),
        },
        run_dir / "completion_manifest.json",
    )
    return result
