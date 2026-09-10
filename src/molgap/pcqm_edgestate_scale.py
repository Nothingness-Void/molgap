"""Durable 500K PCQM EdgeState width/pretraining comparison.

The module streams immutable 50K graph shards so the paired SCNet workers do
not need to materialize the complete 500K role in host memory.  Geometry is
used only as a train-role auxiliary target; the retained encoder is pure 2D.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SUBSET_FORMAT = "molgap-pcqm-edgestate304-500k-subset-v1"
RUN_FORMAT = "molgap-pcqm-edgestate304-500k-run-v1"
TRAIN_ROWS = 500_000
DEVELOPMENT_ROWS = 50_000
EXPECTED_PARAMETER_COUNT = 7_610_945


@dataclass(frozen=True)
class EdgeStateScaleConfig:
    hidden_channels: int = 304
    num_layers: int = 6
    num_heads: int = 4
    edge_state_channels: int = 64
    rwse_dim: int = 16
    dropout: float = 0.05
    batch_size: int = 128
    loader_workers: int = 4
    prefetch_factor: int = 4
    learning_rate: float = 2.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-5
    gradient_clip: float = 1.0
    scratch_epochs: int = 60
    pretrain_epochs: int = 20
    finetune_epochs: int = 40
    mask_rate: float = 0.15
    angle_bins: int = 32
    seed: int = 42

    def validate(self) -> None:
        if self.hidden_channels % self.num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if self.pretrain_epochs + self.finetune_epochs != self.scratch_epochs:
            raise ValueError("paired arms must have equal encoder exposure")
        if self.batch_size != 128:
            raise ValueError("the frozen scale screen requires batch size 128")
        if self.loader_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("loader worker settings must be non-negative")
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError("mask_rate must fall strictly between zero and one")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_graphs(path: Path):
    """Load either legacy graph lists or packed PyG ``(data, slices)`` shards."""
    payload = _torch_load(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple) and len(payload) == 2:
        from torch_geometric.data import InMemoryDataset

        class PackedGraphDataset(InMemoryDataset):
            def __init__(self, packed):
                super().__init__(root=None)
                self.data, self.slices = packed

        return PackedGraphDataset(payload)
    raise TypeError(f"unsupported graph shard payload in {path}: {type(payload)}")


def build_subset_manifest(
    parent_acceptance: Path,
    output_path: Path,
) -> dict:
    """Select train shards 0--9 and shard 10 as a sealed development role."""
    parent = json.loads(parent_acceptance.read_text(encoding="utf-8"))
    if parent.get("status") != "accepted":
        raise RuntimeError("parent geometry cache was not accepted")
    source = {item["path"]: item for item in parent.get("shards", [])}
    selected = []
    for shard_id in range(11):
        parent_path = f"train/train_shard_{shard_id:04d}.pt"
        item = source.get(parent_path)
        if item is None:
            raise RuntimeError(f"missing parent shard: {parent_path}")
        expected_min = shard_id * 50_000
        expected_max = expected_min + 49_999
        if (
            int(item.get("rows", -1)) != 50_000
            or int(item.get("source_idx_min", -1)) != expected_min
            or int(item.get("source_idx_max", -1)) != expected_max
        ):
            raise RuntimeError(f"parent shard identity changed: {parent_path}")
        selected.append(
            {
                "role": "train" if shard_id < 10 else "development",
                "file": parent_path,
                "rows": 50_000,
                "source_idx_min": expected_min,
                "source_idx_max": expected_max,
                "sha256": item["sha256"],
                "bytes": int(item["bytes"]),
            }
        )
    aggregate = hashlib.sha256()
    for item in selected:
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    payload = {
        "format": SUBSET_FORMAT,
        "complete": True,
        "parent_acceptance_sha256": sha256_file(parent_acceptance),
        "parent_format": parent.get("format"),
        "feature_schema": parent.get("feature_schema"),
        "node_feature_dim": parent.get("node_feature_dim"),
        "edge_feature_dim": parent.get("edge_feature_dim"),
        "rwse_dim": parent.get("rwse_dim"),
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "train_source_idx_range": [0, 499_999],
        "development_source_idx_range": [500_000, 549_999],
        "shards": selected,
        "aggregate_sha256": aggregate.hexdigest(),
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output_path, payload)
    return payload


def accept_subset(cache_root: Path, *, verify_payloads: bool = True) -> dict:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "format": manifest.get("format") == SUBSET_FORMAT,
        "complete": manifest.get("complete") is True,
        "feature_schema": manifest.get("feature_schema") == "ogb",
        "node_feature_dim": manifest.get("node_feature_dim") == 9,
        "edge_feature_dim": manifest.get("edge_feature_dim") == 3,
        "rwse_dim": manifest.get("rwse_dim") == 16,
        "train_rows": manifest.get("train_rows") == TRAIN_ROWS,
        "development_rows": manifest.get("development_rows") == DEVELOPMENT_ROWS,
        "official_validation_sealed": manifest.get("official_validation_role_read")
        is False,
        "test_dev_sealed": manifest.get("test_dev_role_read") is False,
    }
    observed = {"train": 0, "development": 0}
    aggregate = hashlib.sha256()
    shard_checks = []
    for item in manifest.get("shards", []):
        path = cache_root / item["file"]
        ok = path.is_file() and path.stat().st_size == int(item["bytes"])
        if ok:
            ok = sha256_file(path) == item["sha256"]
        if ok and verify_payloads:
            graphs = _load_graphs(path)
            ok = len(graphs) == int(item["rows"])
            if ok:
                for graph in (graphs[0], graphs[-1]):
                    ok = ok and graph.x.shape[1] == 9
                    ok = ok and graph.edge_attr.shape[1] == 3
                    ok = ok and graph.random_walk_pe.shape[1] == 16
                    ok = ok and graph.edge_distance.shape[0] == graph.edge_index.shape[1]
                    ok = ok and graph.wedge_angle_cos.shape[0] == graph.wedge_edge_ids.shape[0]
            del graphs
        shard_checks.append(ok)
        if item.get("role") in observed:
            observed[item["role"]] += int(item["rows"])
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    checks["shard_count"] = len(shard_checks) == 11
    checks["all_shards"] = bool(shard_checks) and all(shard_checks)
    checks["role_counts"] = observed == {
        "train": TRAIN_ROWS,
        "development": DEVELOPMENT_ROWS,
    }
    checks["aggregate"] = aggregate.hexdigest() == manifest.get("aggregate_sha256")
    acceptance = {
        "format": "molgap-pcqm-edgestate304-500k-acceptance-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "counts": observed,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(cache_root / "acceptance.json", acceptance)
    if not acceptance["accepted"]:
        raise RuntimeError(f"subset acceptance failed: {checks}")
    return acceptance


def make_model(config: EdgeStateScaleConfig):
    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    return OGBEdgeStateStructuralGPSWrapper(
        in_channels=9,
        edge_dim=3,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        n_targets=1,
        pooling="mean",
        rwse_dim=config.rwse_dim,
        edge_state_channels=config.edge_state_channels,
    )


def parameter_count(config: EdgeStateScaleConfig) -> int:
    return sum(parameter.numel() for parameter in make_model(config).parameters())


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _role_shards(cache_root: Path, manifest: dict, role: str) -> list[Path]:
    return [
        cache_root / item["file"]
        for item in manifest["shards"]
        if item["role"] == role
    ]


def _loader(graphs, config: EdgeStateScaleConfig, *, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    options = {}
    if config.loader_workers:
        options.update(
            num_workers=config.loader_workers,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=False,
        )
    return DataLoader(
        graphs,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        **options,
    )


def _iter_train_batches(cache_root, manifest, config, epoch):
    shard_paths = _role_shards(cache_root, manifest, "train")
    order = list(range(len(shard_paths)))
    random.Random(config.seed + 10_000 * epoch).shuffle(order)
    for position, shard_id in enumerate(order):
        graphs = _load_graphs(shard_paths[shard_id])
        loader = _loader(
            graphs,
            config,
            shuffle=True,
            seed=config.seed + epoch * 101 + position,
        )
        yield from loader
        del loader, graphs


def _target_stats(cache_root: Path, manifest: dict) -> tuple[float, float]:
    import torch

    count = 0
    total = 0.0
    squared = 0.0
    for path in _role_shards(cache_root, manifest, "train"):
        graphs = _load_graphs(path)
        values = torch.tensor([float(graph.y.view(-1)[0]) for graph in graphs])
        count += values.numel()
        total += float(values.sum())
        squared += float(values.square().sum())
        del graphs, values
    mean = total / count
    variance = max(squared / count - mean * mean, 1e-12)
    return mean, math.sqrt(variance)


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def _learning_rate(config: EdgeStateScaleConfig, epoch: int, epochs: int) -> float:
    warmup = min(3, max(1, epochs // 10))
    if epoch < warmup:
        return config.learning_rate * float(epoch + 1) / warmup
    progress = float(epoch - warmup) / max(1, epochs - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _set_optimizer_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _evaluate(model, cache_root, manifest, config, mean, std, device):
    import torch

    model.eval()
    absolute_error = 0.0
    count = 0
    prediction_parts = []
    target_parts = []
    source_idx_parts = []
    with torch.no_grad():
        for path in _role_shards(cache_root, manifest, "development"):
            graphs = _load_graphs(path)
            for batch in _loader(graphs, config, shuffle=False, seed=config.seed):
                batch = batch.to(device, non_blocking=True)
                prediction = _forward(model, batch) * std + mean
                target = batch.y.view(-1).float()
                absolute_error += float((prediction - target).abs().sum())
                count += int(target.numel())
                prediction_parts.append(prediction.cpu())
                target_parts.append(target.cpu())
                source_idx = getattr(batch, "source_idx", getattr(batch, "row_index", None))
                if source_idx is None:
                    raise RuntimeError("development graph has no source identity")
                source_idx_parts.append(source_idx.view(-1).cpu())
            del graphs
    return {
        "mae_eV": absolute_error / count,
        "count": count,
        "prediction": torch.cat(prediction_parts),
        "target": torch.cat(target_parts),
        "source_idx": torch.cat(source_idx_parts),
    }


class RelationHeads:
    """Training-only local heads for masked atom, bond, and angle relations."""

    def __init__(self, config: EdgeStateScaleConfig):
        import torch.nn as nn
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.atom_heads = nn.ModuleList(
            nn.Linear(config.hidden_channels, classes)
            for classes in get_atom_feature_dims()
        )
        self.bond_heads = nn.ModuleList(
            nn.Linear(config.edge_state_channels, classes)
            for classes in get_bond_feature_dims()
        )
        relation_channels = config.hidden_channels + 2 * config.edge_state_channels
        self.angle_head = nn.Sequential(
            nn.LayerNorm(relation_channels),
            nn.Linear(relation_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Linear(config.hidden_channels, config.angle_bins),
        )
        self.module = nn.ModuleDict(
            {
                "atom_heads": self.atom_heads,
                "bond_heads": self.bond_heads,
                "angle_head": self.angle_head,
            }
        )


def _undirected_edge_mask(batch, generator, rate: float):
    import torch

    if batch.edge_index.shape[1] == 0:
        return torch.zeros(0, dtype=torch.bool)
    source, target = batch.edge_index.cpu()
    node_count = int(batch.x.shape[0])
    keys = torch.minimum(source, target) * node_count + torch.maximum(source, target)
    _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    selected = torch.rand(int(inverse.max()) + 1, generator=generator) < rate
    mask = selected[inverse]
    if not bool(mask.any()):
        mask[inverse == inverse[0]] = True
    return mask


def _relation_loss(model, heads: RelationHeads, batch, config, generator):
    import torch
    import torch.nn.functional as functional

    original_x = batch.x.clone()
    original_edge = batch.edge_attr.clone()
    node_mask = torch.rand(batch.x.shape[0], generator=generator) < config.mask_rate
    if not bool(node_mask.any()):
        node_mask[0] = True
    edge_mask = _undirected_edge_mask(batch, generator, config.mask_rate)
    batch.x[node_mask] = 0
    if edge_mask.numel():
        batch.edge_attr[edge_mask] = 0
    device = next(model.parameters()).device
    batch = batch.to(device, non_blocking=True)
    node_mask = node_mask.to(device)
    edge_mask = edge_mask.to(device)
    original_x = original_x.to(device)
    original_edge = original_edge.to(device)
    node_state, edge_state = model._encode_states(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    atom_loss = sum(
        functional.cross_entropy(head(node_state[node_mask]), original_x[node_mask, column])
        for column, head in enumerate(heads.atom_heads)
    ) / len(heads.atom_heads)
    bond_loss = atom_loss.new_zeros(())
    if edge_mask.numel() and bool(edge_mask.any()):
        bond_loss = sum(
            functional.cross_entropy(
                head(edge_state[edge_mask]), original_edge[edge_mask, column]
            )
            for column, head in enumerate(heads.bond_heads)
        ) / len(heads.bond_heads)
    wedge_ids = batch.wedge_edge_ids.long()
    first = wedge_ids[:, 0]
    second = wedge_ids[:, 1]
    centers = batch.edge_index[1, first]
    valid_graph = batch.geometry_valid.view(-1).bool()
    valid_wedge = valid_graph[batch.batch[centers]]
    angle_loss = atom_loss.new_zeros(())
    if bool(valid_wedge.any()):
        relation = torch.cat(
            [node_state[centers], edge_state[first], edge_state[second]], dim=-1
        )[valid_wedge]
        angle = batch.wedge_angle_cos.view(-1)[valid_wedge]
        angle_target = (((angle + 1.0) * 0.5) * config.angle_bins).floor().long()
        angle_target = angle_target.clamp(0, config.angle_bins - 1)
        angle_loss = functional.cross_entropy(heads.angle_head(relation), angle_target)
    return atom_loss + bond_loss + angle_loss, {
        "atom": atom_loss,
        "bond": bond_loss,
        "angle": angle_loss,
    }


def _checkpoint_payload(
    *, config, role, stage, epoch, model, optimizer, trace, initial_hash,
    cache_sha256, heads=None,
):
    import numpy as np
    import torch

    payload = {
        "format": RUN_FORMAT,
        "config": asdict(config),
        "role": role,
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trace": trace,
        "initial_encoder_sha256": initial_hash,
        "cache_aggregate_sha256": cache_sha256,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state": torch.cuda.get_rng_state_all(),
    }
    if heads is not None:
        payload["heads"] = heads.module.state_dict()
    return payload


def _load_resume(path, *, config, role, stage, model, optimizer, heads=None):
    import numpy as np
    import torch

    if not path.is_file():
        return 0, []
    state = _torch_load(path)
    if (
        state.get("format") != RUN_FORMAT
        or state.get("config") != asdict(config)
        or state.get("role") != role
        or state.get("stage") != stage
    ):
        raise RuntimeError(f"resume contract changed: {path}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if heads is not None:
        heads.module.load_state_dict(state["heads"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["accelerator_rng_state"])
    return int(state["epoch"]) + 1, list(state["trace"])


def _train_gap(
    model, cache_root, manifest, config, output, *, role, stage, epochs,
    initial_hash, mean, std,
):
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda:0")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint = output / f"{stage}_last.pt"
    start_epoch, trace = _load_resume(
        checkpoint,
        config=config,
        role=role,
        stage=stage,
        model=model,
        optimizer=optimizer,
    )
    best = min((row["development_mae_eV"] for row in trace), default=float("inf"))
    best_epoch = next(
        (row["epoch"] for row in trace if row["development_mae_eV"] == best), -1
    )
    mean_tensor = torch.tensor(mean, device=device)
    std_tensor = torch.tensor(std, device=device)
    for epoch in range(start_epoch, epochs):
        lr = _learning_rate(config, epoch, epochs)
        _set_optimizer_lr(optimizer, lr)
        model.train()
        started = time.perf_counter()
        absolute = 0.0
        count = 0
        torch.cuda.reset_peak_memory_stats(device)
        for batch in _iter_train_batches(cache_root, manifest, config, epoch):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1).float() - mean_tensor) / std_tensor
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite Gap loss at {stage} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            absolute += float(
                (prediction.detach() * std_tensor + mean_tensor - batch.y.view(-1))
                .abs()
                .sum()
            )
            count += int(batch.y.numel())
        development = _evaluate(
            model, cache_root, manifest, config, mean_tensor, std_tensor, device
        )
        elapsed = time.perf_counter() - started
        improved = development["mae_eV"] < best
        if improved:
            best = development["mae_eV"]
            best_epoch = epoch
            atomic_torch_save(
                output / f"{stage}_best.pt",
                {
                    "format": RUN_FORMAT,
                    "config": asdict(config),
                    "role": role,
                    "stage": stage,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "target_mean": mean,
                    "target_std": std,
                    "cache_aggregate_sha256": manifest["aggregate_sha256"],
                },
            )
            atomic_torch_save(
                output / f"{stage}_development.pt",
                {
                    "prediction_eV": development["prediction"],
                    "target_eV": development["target"],
                    "source_idx": development["source_idx"],
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
        row = {
            "epoch": epoch,
            "train_mae_eV": absolute / count,
            "development_mae_eV": development["mae_eV"],
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_ROWS / elapsed,
            "learning_rate": lr,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "improved": improved,
        }
        trace.append(row)
        atomic_json(output / f"{stage}_trace.json", {"epochs": trace})
        atomic_torch_save(
            checkpoint,
            _checkpoint_payload(
                config=config,
                role=role,
                stage=stage,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                trace=trace,
                initial_hash=initial_hash,
                cache_sha256=manifest["aggregate_sha256"],
            ),
        )
        print(
            f"{role}/{stage} ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"dev={row['development_mae_eV']:.6f}eV {elapsed:.1f}s "
            f"peak={row['peak_memory_bytes'] / 2**30:.2f}GiB"
            f"{' *' if improved else ''}",
            flush=True,
        )
    best_state = _torch_load(output / f"{stage}_best.pt")
    model.load_state_dict(best_state["model"])
    return model, {
        "epochs_completed": len(trace),
        "best_epoch": best_epoch,
        "best_development_mae_eV": best,
        "best_sha256": sha256_file(output / f"{stage}_best.pt"),
        "last_sha256": sha256_file(checkpoint),
        "trace": trace,
    }


def _pretrain(model, cache_root, manifest, config, output, initial_hash):
    import torch

    device = torch.device("cuda:0")
    model = model.to(device)
    heads = RelationHeads(config)
    heads.module = heads.module.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(heads.module.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint = output / "relation_pretrain_last.pt"
    start_epoch, trace = _load_resume(
        checkpoint,
        config=config,
        role="pretrained",
        stage="relation_pretrain",
        model=model,
        optimizer=optimizer,
        heads=heads,
    )
    for epoch in range(start_epoch, config.pretrain_epochs):
        lr = _learning_rate(config, epoch, config.pretrain_epochs)
        _set_optimizer_lr(optimizer, lr)
        model.train()
        heads.module.train()
        generator = torch.Generator().manual_seed(config.seed + 31_337 * (epoch + 1))
        totals = {"loss": 0.0, "atom": 0.0, "bond": 0.0, "angle": 0.0}
        batches = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for batch in _iter_train_batches(cache_root, manifest, config, epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _relation_loss(model, heads, batch, config, generator)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite relation loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.module.parameters()),
                config.gradient_clip,
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in parts.items():
                totals[name] += float(value.detach())
            batches += 1
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            **{name: value / batches for name, value in totals.items()},
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_ROWS / elapsed,
            "learning_rate": lr,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        trace.append(row)
        atomic_json(output / "relation_pretrain_trace.json", {"epochs": trace})
        atomic_torch_save(
            checkpoint,
            _checkpoint_payload(
                config=config,
                role="pretrained",
                stage="relation_pretrain",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                trace=trace,
                initial_hash=initial_hash,
                cache_sha256=manifest["aggregate_sha256"],
                heads=heads,
            ),
        )
        print(
            f"pretrained/relation_pretrain ep{epoch:02d} loss={row['loss']:.6f} "
            f"atom={row['atom']:.6f} bond={row['bond']:.6f} "
            f"angle={row['angle']:.6f} {elapsed:.1f}s",
            flush=True,
        )
    result = {
        "epochs_completed": len(trace),
        "last_sha256": sha256_file(checkpoint),
        "final_encoder_sha256": state_sha256(model),
        "gap_labels_read": False,
        "trace": trace,
    }
    del heads
    return model, result


def run_preflight(cache_root: Path, output: Path, config: EdgeStateScaleConfig) -> dict:
    import torch

    config.validate()
    acceptance = accept_subset(cache_root, verify_payloads=True)
    set_seed(config.seed)
    model = make_model(config).to("cuda:0")
    params = sum(parameter.numel() for parameter in model.parameters())
    if params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"model parameter count changed: {params} != {EXPECTED_PARAMETER_COUNT}"
        )
    first_shard = _role_shards(
        cache_root,
        json.loads((cache_root / "manifest.json").read_text(encoding="utf-8")),
        "train",
    )[0]
    graphs = _load_graphs(first_shard)
    batch = next(iter(_loader(graphs, config, shuffle=False, seed=config.seed)))
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    batch = batch.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    prediction = _forward(model, batch)
    loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1).float())
    loss.backward()
    optimizer.step()
    scratch_peak = int(torch.cuda.max_memory_allocated(device))
    del optimizer, model, batch
    torch.cuda.empty_cache()

    set_seed(config.seed)
    model = make_model(config).to(device)
    heads = RelationHeads(config)
    heads.module = heads.module.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(heads.module.parameters()),
        lr=config.learning_rate,
    )
    batch = next(iter(_loader(graphs, config, shuffle=False, seed=config.seed)))
    generator = torch.Generator().manual_seed(config.seed + 31_337)
    torch.cuda.reset_peak_memory_stats(device)
    relation_loss, parts = _relation_loss(model, heads, batch, config, generator)
    relation_loss.backward()
    optimizer.step()
    pretrained_peak = int(torch.cuda.max_memory_allocated(device))
    payload = {
        "format": "molgap-pcqm-edgestate304-500k-preflight-v1",
        "accepted": bool(torch.isfinite(loss) and torch.isfinite(relation_loss)),
        "config": asdict(config),
        "parameter_count": params,
        "batch_size": config.batch_size,
        "scratch_loss": float(loss.detach()),
        "relation_loss": float(relation_loss.detach()),
        "relation_parts": {name: float(value.detach()) for name, value in parts.items()},
        "scratch_peak_memory_bytes": scratch_peak,
        "pretrained_peak_memory_bytes": pretrained_peak,
        "cache_acceptance": acceptance,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "preflight.json", payload)
    if not payload["accepted"]:
        raise RuntimeError("preflight produced non-finite values")
    return payload


def run_worker(
    role: str,
    cache_root: Path,
    output: Path,
    config: EdgeStateScaleConfig,
    *,
    source_commit: str,
) -> dict:
    import torch

    config.validate()
    if role not in {"scratch", "pretrained"}:
        raise ValueError(f"unsupported role: {role}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SCNet worker requires exactly one visible DCU")
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        complete = json.loads(completion_path.read_text(encoding="utf-8"))
        if complete.get("complete") is True and complete.get("config") == asdict(config):
            return complete
        raise RuntimeError("incompatible completion manifest already exists")
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((cache_root / "acceptance.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SUBSET_FORMAT or acceptance.get("accepted") is not True:
        raise RuntimeError("500K subset has not passed acceptance")
    set_seed(config.seed)
    model = make_model(config)
    params = sum(parameter.numel() for parameter in model.parameters())
    if params != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"model parameter count changed: {params} != {EXPECTED_PARAMETER_COUNT}"
        )
    initial_hash = state_sha256(model)
    mean, std = _target_stats(cache_root, manifest)
    started = time.perf_counter()
    pretraining = None
    if role == "pretrained":
        model, pretraining = _pretrain(
            model, cache_root, manifest, config, output, initial_hash
        )
        model, gap = _train_gap(
            model,
            cache_root,
            manifest,
            config,
            output,
            role=role,
            stage="finetune_gap",
            epochs=config.finetune_epochs,
            initial_hash=initial_hash,
            mean=mean,
            std=std,
        )
    else:
        model, gap = _train_gap(
            model,
            cache_root,
            manifest,
            config,
            output,
            role=role,
            stage="scratch_gap",
            epochs=config.scratch_epochs,
            initial_hash=initial_hash,
            mean=mean,
            std=std,
        )
    payload = {
        "format": RUN_FORMAT,
        "complete": True,
        "role": role,
        "source_commit": source_commit,
        "config": asdict(config),
        "architecture": "OGB EdgeState GPS6 width304",
        "parameter_count": params,
        "initial_encoder_sha256": initial_hash,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "target_mean": mean,
        "target_std": std,
        "pretraining": pretraining,
        "gap": gap,
        "elapsed_s": time.perf_counter() - started,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "metrics.json", payload)
    atomic_json(completion_path, payload)
    return payload
