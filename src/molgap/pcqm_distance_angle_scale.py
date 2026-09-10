"""Durable paired 500K PCQM distance/angle bottom-fusion screen.

Both arms stream the same immutable 50K graph shards.  The only experimental
variable is whether ETKDGv3+MMFF94s bond distances and wedge angles are fused
into the accepted OGB EdgeState Structural GPS9 backbone.
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
RUN_FORMAT = "molgap-pcqm-distance-angle-500k-run-v1"
TRAIN_ROWS = 500_000
DEVELOPMENT_ROWS = 50_000
BASELINE = "ogb_edge_state_structural_gps9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_gps9"
ARMS = (BASELINE, CANDIDATE)
EXPECTED_PARAMETER_COUNTS = {
    BASELINE: 4_771_073,
    CANDIDATE: 4_891_057,
}


@dataclass(frozen=True)
class DistanceAngleScaleConfig:
    hidden_channels: int = 192
    num_layers: int = 9
    num_heads: int = 4
    edge_state_channels: int = 64
    wedge_channels: int = 16
    geometry_basis_channels: int = 16
    rwse_dim: int = 16
    dropout: float = 0.1
    batch_size: int = 128
    loader_workers: int = 4
    prefetch_factor: int = 4
    learning_rate: float = 1.6e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-6
    gradient_clip: float = 1.0
    epochs: int = 60
    seed: int = 42

    def validate(self) -> None:
        if self.hidden_channels % self.num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if self.batch_size != 128:
            raise ValueError("the frozen scale screen requires batch size 128")
        if self.loader_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("loader worker settings must be non-negative")
        if self.epochs != 60:
            raise ValueError("the frozen paired screen requires 60 direct-Gap passes")


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


def make_model(arm: str, config: DistanceAngleScaleConfig):
    from .pcqm_gap_architecture import make_pcqm_gap_encoder

    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    expected = {
        "hidden_channels": 192,
        "num_layers": 9,
        "num_heads": 4,
        "edge_state_channels": 64,
        "wedge_channels": 16,
        "geometry_basis_channels": 16,
        "rwse_dim": 16,
        "dropout": 0.1,
    }
    observed = {name: getattr(config, name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"architecture contract changed: {observed}")
    return make_pcqm_gap_encoder(arm)


def parameter_count(arm: str, config: DistanceAngleScaleConfig) -> int:
    return sum(parameter.numel() for parameter in make_model(arm, config).parameters())


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


def _loader(graphs, config: DistanceAngleScaleConfig, *, shuffle: bool, seed: int):
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


def _forward(model, batch, arm: str):
    if arm == CANDIDATE:
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.wedge_edge_ids,
            batch.edge_distance,
            batch.wedge_angle_cos,
            batch.geometry_valid,
        ).view(-1)
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def _learning_rate(config: DistanceAngleScaleConfig, epoch: int, epochs: int) -> float:
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


def _evaluate(model, cache_root, manifest, config, mean, std, device, *, arm):
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
                prediction = _forward(model, batch, arm) * std + mean
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
            prediction = _forward(model, batch, role)
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
            model,
            cache_root,
            manifest,
            config,
            mean_tensor,
            std_tensor,
            device,
            arm=role,
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


def run_preflight(
    cache_root: Path, output: Path, config: DistanceAngleScaleConfig
) -> dict:
    import torch

    config.validate()
    acceptance = accept_subset(cache_root, verify_payloads=True)
    first_shard = _role_shards(
        cache_root,
        json.loads((cache_root / "manifest.json").read_text(encoding="utf-8")),
        "train",
    )[0]
    graphs = _load_graphs(first_shard)
    device = torch.device("cuda:0")
    arm_results = {}
    baseline_state = None
    for arm in ARMS:
        set_seed(config.seed)
        model = make_model(arm, config)
        params = sum(parameter.numel() for parameter in model.parameters())
        if params != EXPECTED_PARAMETER_COUNTS[arm]:
            raise RuntimeError(
                f"{arm} parameter count changed: {params} != "
                f"{EXPECTED_PARAMETER_COUNTS[arm]}"
            )
        state = model.state_dict()
        if arm == BASELINE:
            baseline_state = {
                name: value.detach().clone() for name, value in state.items()
            }
        else:
            mismatches = [
                name
                for name in sorted(set(baseline_state).intersection(state))
                if not torch.equal(baseline_state[name], state[name])
            ]
            if mismatches:
                raise RuntimeError(
                    f"shared initialization changed for {arm}: {mismatches[:10]}"
                )
        model = model.to(device)
        batch = next(iter(_loader(graphs, config, shuffle=False, seed=config.seed)))
        batch = batch.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        torch.cuda.reset_peak_memory_stats(device)
        prediction = _forward(model, batch, arm)
        loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1).float())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{arm} preflight produced non-finite loss")
        loss.backward()
        optimizer.step()
        arm_results[arm] = {
            "parameter_count": params,
            "loss": float(loss.detach()),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        del optimizer, model, batch
        torch.cuda.empty_cache()
    payload = {
        "format": "molgap-pcqm-distance-angle-500k-preflight-v1",
        "accepted": True,
        "config": asdict(config),
        "batch_size": config.batch_size,
        "arms": arm_results,
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
    config: DistanceAngleScaleConfig,
    *,
    source_commit: str,
) -> dict:
    import torch

    config.validate()
    if role not in ARMS:
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
    model = make_model(role, config)
    params = sum(parameter.numel() for parameter in model.parameters())
    if params != EXPECTED_PARAMETER_COUNTS[role]:
        raise RuntimeError(
            f"model parameter count changed: {params} != "
            f"{EXPECTED_PARAMETER_COUNTS[role]}"
        )
    initial_hash = state_sha256(model)
    mean, std = _target_stats(cache_root, manifest)
    started = time.perf_counter()
    model, gap = _train_gap(
        model,
        cache_root,
        manifest,
        config,
        output,
        role=role,
        stage="direct_gap",
        epochs=config.epochs,
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
        "architecture": role,
        "parameter_count": params,
        "initial_encoder_sha256": initial_hash,
        "cache_aggregate_sha256": manifest["aggregate_sha256"],
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "target_mean": mean,
        "target_std": std,
        "gap": gap,
        "elapsed_s": time.perf_counter() - started,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "metrics.json", payload)
    atomic_json(completion_path, payload)
    return payload


def accept_pair(output_root: Path) -> dict:
    """Mechanically accept both completed arms and recompute the paired delta."""
    import torch

    records = {}
    for arm in ARMS:
        arm_root = output_root / arm
        completion_path = arm_root / "completion_manifest.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        gap = completion.get("gap", {})
        best_path = arm_root / "direct_gap_best.pt"
        last_path = arm_root / "direct_gap_last.pt"
        development_path = arm_root / "direct_gap_development.pt"
        prediction_payload = _torch_load(development_path)
        prediction = prediction_payload["prediction_eV"].view(-1).float()
        target = prediction_payload["target_eV"].view(-1).float()
        source_idx = prediction_payload["source_idx"].view(-1).long()
        recomputed_mae = float((prediction - target).abs().mean())
        checks = {
            "complete": completion.get("complete") is True,
            "format": completion.get("format") == RUN_FORMAT,
            "role": completion.get("role") == arm,
            "architecture": completion.get("architecture") == arm,
            "parameter_count": completion.get("parameter_count")
            == EXPECTED_PARAMETER_COUNTS[arm],
            "train_rows": completion.get("train_rows") == TRAIN_ROWS,
            "development_rows": completion.get("development_rows")
            == DEVELOPMENT_ROWS,
            "epochs": gap.get("epochs_completed") == 60,
            "best_exists": best_path.is_file(),
            "last_exists": last_path.is_file(),
            "development_exists": development_path.is_file(),
            "best_sha256": sha256_file(best_path) == gap.get("best_sha256"),
            "last_sha256": sha256_file(last_path) == gap.get("last_sha256"),
            "prediction_count": prediction.numel() == DEVELOPMENT_ROWS,
            "target_count": target.numel() == DEVELOPMENT_ROWS,
            "source_idx_count": source_idx.numel() == DEVELOPMENT_ROWS,
            "source_idx_order": torch.equal(
                source_idx, torch.arange(500_000, 550_000, dtype=torch.long)
            ),
            "finite_prediction": bool(torch.isfinite(prediction).all()),
            "finite_target": bool(torch.isfinite(target).all()),
            "mae_recomputed": math.isclose(
                recomputed_mae,
                float(gap.get("best_development_mae_eV", float("nan"))),
                rel_tol=0.0,
                abs_tol=1e-7,
            ),
            "official_validation_sealed": completion.get(
                "official_validation_role_read"
            )
            is False,
            "test_dev_sealed": completion.get("test_dev_role_read") is False,
        }
        records[arm] = {
            "accepted": all(checks.values()),
            "checks": checks,
            "best_development_mae_eV": recomputed_mae,
            "best_epoch": gap.get("best_epoch"),
            "source_commit": completion.get("source_commit"),
            "config": completion.get("config"),
            "cache_aggregate_sha256": completion.get("cache_aggregate_sha256"),
            "best_sha256": gap.get("best_sha256"),
            "last_sha256": gap.get("last_sha256"),
        }
    baseline = records[BASELINE]
    candidate = records[CANDIDATE]
    paired_checks = {
        "both_accepted": baseline["accepted"] and candidate["accepted"],
        "same_source_commit": baseline["source_commit"]
        == candidate["source_commit"],
        "same_config": baseline["config"] == candidate["config"],
        "same_cache": baseline["cache_aggregate_sha256"]
        == candidate["cache_aggregate_sha256"],
    }
    delta = (
        candidate["best_development_mae_eV"]
        - baseline["best_development_mae_eV"]
    )
    payload = {
        "format": "molgap-pcqm-distance-angle-500k-paired-acceptance-v1",
        "accepted": all(paired_checks.values()),
        "paired_checks": paired_checks,
        "arms": records,
        "candidate_minus_baseline_mae_eV": delta,
        "nomination_threshold_eV": -0.001,
        "nominated": all(paired_checks.values()) and delta <= -0.001,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output_root / "paired_acceptance.json", payload)
    if not payload["accepted"]:
        raise RuntimeError(f"paired result acceptance failed: {paired_checks}")
    return payload
