"""Train-role-only graph-alignment PE screen on the transferable QM9 contract."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from .screen_policy import validate_paired_screen_contract, validate_screen_arm


TRAIN_ROWS = 30_000
VALIDATION_ROWS = 3_000
SEED = 42
BATCH_SIZE = 128
RWSE_DIM = 16
GAPE_DIM = 32
GAPE_PRETRAIN_EPOCHS = 10
GAP_EPOCHS = 40
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
GAPE_LEARNING_RATE = 1e-3
EDGE_DROP_RATE = 0.30
TEMPERATURE = 0.10
MIN_GAIN_EV = 0.003
TASK_ID = "qm9-gape-lite-s42-v1"
PLATFORM_ID = "kaggle2"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TopologyAlignmentGenerator:
    """Factory namespace that keeps torch imports remote-only."""

    @staticmethod
    def make(channels: int = GAPE_DIM, layers: int = 3):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GATConv

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(1, channels)
                self.convs = nn.ModuleList(
                    GATConv(
                        channels,
                        channels // 4,
                        heads=4,
                        concat=True,
                        add_self_loops=True,
                    )
                    for _ in range(layers)
                )
                self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(layers))

            def forward(self, node_count, edge_index):
                values = torch.ones((node_count, 1), device=edge_index.device)
                hidden = self.input(values)
                for conv, norm in zip(self.convs, self.norms):
                    hidden = norm(hidden + torch.nn.functional.silu(conv(hidden, edge_index)))
                return torch.nn.functional.normalize(hidden, dim=-1)

        return Generator()


def make_augmented_encoder():
    import torch
    import torch.nn as nn

    from .pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

    class GAPEEdgeStateGPS(OGBEdgeStateStructuralGPSWrapper):
        def __init__(self):
            super().__init__(
                in_channels=9,
                edge_dim=3,
                hidden_channels=192,
                num_layers=9,
                num_heads=4,
                dropout=0.05,
                n_targets=1,
                pooling="mean",
                rwse_dim=RWSE_DIM,
                edge_state_channels=64,
            )
            self.gape_projection = nn.Sequential(
                nn.LayerNorm(GAPE_DIM),
                nn.Linear(GAPE_DIM, 192, bias=False),
            )

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe, gape_pe):
            return self.head(
                self.encode(x, edge_index, edge_attr, batch, random_walk_pe, gape_pe)
            )

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe, gape_pe):
            expected_rwse = (x.shape[0], RWSE_DIM)
            expected_gape = (x.shape[0], GAPE_DIM)
            if tuple(random_walk_pe.shape) != expected_rwse:
                raise ValueError(f"RWSE shape changed: {tuple(random_walk_pe.shape)}")
            if tuple(gape_pe.shape) != expected_gape:
                raise ValueError(f"GAPE shape changed: {tuple(gape_pe.shape)}")
            if not torch.isfinite(gape_pe).all():
                raise ValueError("GAPE contains non-finite values")
            h = self._embed_nodes(x)
            h = h + self.rwse_encoder(random_walk_pe.float())
            h = h + self.gape_projection(gape_pe.float())
            edge_state = self._embed_edges(edge_attr)
            for edge_update, conv in zip(self.edge_updates, self.convs):
                edge_state = edge_update(h, edge_index, edge_state)
                h = conv(h, edge_index, batch, edge_attr=edge_state)
            return self._pool(h, batch)

    return GAPEEdgeStateGPS()


def make_baseline_encoder():
    from .qm9_local_hierarchy import make_encoder

    return make_encoder()


def make_loader(graphs, *, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def corrupt_edges(edge_index):
    import torch

    if edge_index.shape[1] == 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= EDGE_DROP_RATE
    if not bool(keep.any()):
        keep[torch.randint(edge_index.shape[1], (1,), device=edge_index.device)] = True
    return edge_index[:, keep]


def alignment_loss(left, right, batch, *, matched: bool):
    import torch
    import torch.nn.functional as functional

    losses = []
    graph_count = int(batch.max().item()) + 1
    for graph_id in range(graph_count):
        indices = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
        similarity = left[indices] @ right[indices].T / TEMPERATURE
        row_target = torch.arange(indices.numel(), device=similarity.device)
        column_target = row_target
        if not matched and row_target.numel() > 1:
            row_target = torch.roll(row_target, shifts=1)
            column_target = torch.roll(column_target, shifts=-1)
        losses.append(
            0.5
            * (
                functional.cross_entropy(similarity, row_target)
                + functional.cross_entropy(similarity.T, column_target)
            )
        )
    return torch.stack(losses).mean()


def pretrain_generator(graphs, output_dir: Path, *, matched: bool, source_commit: str):
    import torch

    set_seed(SEED)
    device = torch.device("cuda")
    generator = TopologyAlignmentGenerator.make().to(device)
    initial_sha = state_sha256(generator)
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=GAPE_LEARNING_RATE, weight_decay=0.0
    )
    loader = make_loader(graphs, shuffle=True, seed=SEED)
    trace = []
    for epoch in range(GAPE_PRETRAIN_EPOCHS):
        generator.train()
        total = 0.0
        batches = 0
        started = time.perf_counter()
        for graph_batch in loader:
            graph_batch = graph_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            left = generator(
                graph_batch.num_nodes, corrupt_edges(graph_batch.edge_index)
            )
            right = generator(
                graph_batch.num_nodes, corrupt_edges(graph_batch.edge_index)
            )
            loss = alignment_loss(left, right, graph_batch.batch, matched=matched)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "alignment_loss": total / batches,
            "seconds": time.perf_counter() - started,
        }
        trace.append(row)
        atomic_torch_save(
            output_dir / "last_checkpoint.pt",
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "matched": matched,
                "batch_size": BATCH_SIZE,
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"gape_{'matched' if matched else 'shuffled'} ep{epoch:02d} "
            f"loss={row['alignment_loss']:.6f} {row['seconds']:.1f}s",
            flush=True,
        )
    return generator, {
        "initial_generator_sha256": initial_sha,
        "final_generator_sha256": state_sha256(generator),
        "generator_parameter_count": sum(
            parameter.numel() for parameter in generator.parameters()
        ),
        "checkpoint_sha256": sha256_file(output_dir / "last_checkpoint.pt"),
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "matched_correspondence": matched,
    }


def attach_gape(graphs, generator) -> str:
    import torch

    device = next(generator.parameters()).device
    generator.eval()
    digest = hashlib.sha256()
    cursor = 0
    with torch.no_grad():
        for graph_batch in make_loader(graphs, shuffle=False, seed=SEED):
            graph_batch = graph_batch.to(device)
            values = generator(graph_batch.num_nodes, graph_batch.edge_index).cpu()
            pointers = graph_batch.ptr.cpu().tolist()
            for left, right in zip(pointers[:-1], pointers[1:]):
                value = values[left:right].contiguous()
                graphs[cursor].gape_pe = value
                digest.update(value.numpy().tobytes())
                cursor += 1
    if cursor != len(graphs):
        raise RuntimeError("GAPE materialization count changed")
    return digest.hexdigest()


def target_stats(graphs):
    import torch

    target = torch.stack([graph.y.view(()) for graph in graphs])
    return target.mean(), target.std().clamp_min(1e-6)


def forward_gap(model, batch, *, augmented: bool):
    if augmented:
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.gape_pe,
        ).view(-1)
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def evaluate(model, loader, mean, std, device, *, augmented: bool):
    import torch

    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = forward_gap(model, batch, augmented=augmented) * std + mean
            targets.append(batch.y.view(-1).cpu())
            predictions.append(prediction.cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    return float((prediction - target).abs().mean()), target, prediction


def train_gap(
    model,
    roles,
    output_dir: Path,
    *,
    augmented: bool,
    source_commit: str,
    cache_sha256: str,
):
    import torch
    import torch.nn.functional as functional

    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model.to(device)
    initial_sha = state_sha256(model)
    mean, std = target_stats(roles["train"])
    mean, std = mean.to(device), std.to(device)
    train_loader = make_loader(roles["train"], shuffle=True, seed=SEED)
    validation_loader = make_loader(roles["validation"], shuffle=False, seed=SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=GAP_EPOCHS, eta_min=1e-6
    )
    best = float("inf")
    best_epoch = -1
    trace = []
    for epoch in range(GAP_EPOCHS):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_gap(model, batch, augmented=augmented)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float((prediction.detach() - target).abs().sum())
            rows += int(target.numel())
        validation_mae, target_eV, prediction_eV = evaluate(
            model,
            validation_loader,
            mean,
            std,
            device,
            augmented=augmented,
        )
        improved = validation_mae < best
        if improved:
            best = validation_mae
            best_epoch = epoch
            atomic_torch_save(output_dir / "best_model.pt", model.state_dict())
            atomic_torch_save(
                output_dir / "best_validation_payload.pt",
                {"target_eV": target_eV, "prediction_eV": prediction_eV},
            )
        row = {
            "epoch": epoch,
            "train_normalized_mae": absolute / rows,
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
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        )
        atomic_json(output_dir / "trace.json", {"epochs": trace})
        print(
            f"{output_dir.name} ep{epoch:02d} train={row['train_normalized_mae']:.6f} "
            f"val={validation_mae:.6f}eV {row['seconds']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
    return {
        "initial_model_sha256": initial_sha,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "best_model_sha256": sha256_file(output_dir / "best_model.pt"),
        "payload_sha256": sha256_file(output_dir / "best_validation_payload.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "last_checkpoint.pt"),
    }


def arm_contract(arm: str, *, accelerator: str, split_fingerprint: str) -> dict:
    return {
        "arm": arm,
        "task_id": TASK_ID,
        "platform_id": PLATFORM_ID,
        "accelerator": accelerator,
        "data_role_fingerprint": split_fingerprint,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine40-eta1e-6",
        "sample_exposure": "qm9-train30000-gap40",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def run_worker(
    role: str,
    cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    cache_sha256: str,
):
    import torch

    from .qm9_local_hierarchy import load_cache

    if role not in {"baseline", "gape"}:
        raise ValueError(f"Unknown worker role: {role}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker requires exactly one visible GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, manifest = load_cache(cache_root, cache_sha256)
    accelerator = torch.cuda.get_device_name(0)
    if role == "baseline":
        set_seed(SEED)
        result = {
            "arm": "baseline",
            "training": train_gap(
                make_baseline_encoder(),
                roles,
                output_root / "baseline",
                augmented=False,
                source_commit=source_commit,
                cache_sha256=cache_sha256,
            ),
            "contract": arm_contract(
                "baseline", accelerator=accelerator,
                split_fingerprint=manifest["split_fingerprint"]
            ),
            "test_role_read": False,
        }
        atomic_json(output_root / "baseline_worker.json", result)
        return result

    generators = {}
    generator_reports = {}
    initial_generator_sha = None
    for name, matched in (("shuffled_control", False), ("matched_gape", True)):
        generator, report = pretrain_generator(
            roles["train"],
            output_root / f"{name}_pretrain",
            matched=matched,
            source_commit=source_commit,
        )
        if initial_generator_sha is None:
            initial_generator_sha = report["initial_generator_sha256"]
        elif report["initial_generator_sha256"] != initial_generator_sha:
            raise RuntimeError("GAPE generator initialization changed between controls")
        generators[name] = generator
        generator_reports[name] = report

    results = {}
    initial_augmented_sha = None
    for name in ("shuffled_control", "matched_gape"):
        train_pe_sha = attach_gape(roles["train"], generators[name])
        validation_pe_sha = attach_gape(roles["validation"], generators[name])
        set_seed(SEED)
        model = make_augmented_encoder()
        observed_initial = state_sha256(model)
        if initial_augmented_sha is None:
            initial_augmented_sha = observed_initial
        elif observed_initial != initial_augmented_sha:
            raise RuntimeError("Augmented encoder initialization changed between arms")
        results[name] = {
            "generator": generator_reports[name],
            "train_gape_sha256": train_pe_sha,
            "validation_gape_sha256": validation_pe_sha,
            "training": train_gap(
                model,
                roles,
                output_root / name,
                augmented=True,
                source_commit=source_commit,
                cache_sha256=cache_sha256,
            ),
            "contract": arm_contract(
                name,
                accelerator=accelerator,
                split_fingerprint=manifest["split_fingerprint"],
            ),
            "test_role_read": False,
        }
        del model
        torch.cuda.empty_cache()
    result = {
        "arm": "gape",
        "initial_generator_sha256": initial_generator_sha,
        "initial_augmented_model_sha256": initial_augmented_sha,
        "results": results,
        "test_role_read": False,
    }
    atomic_json(output_root / "gape_worker.json", result)
    return result


def aggregate(output_root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    baseline = json.loads((output_root / "baseline_worker.json").read_text())
    gape = json.loads((output_root / "gape_worker.json").read_text())
    results = {
        "baseline": baseline,
        **gape["results"],
    }
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    baseline_mae = results["baseline"]["training"]["validation_gap_mae_eV"]
    control_mae = results["shuffled_control"]["training"]["validation_gap_mae_eV"]
    candidate_mae = results["matched_gape"]["training"]["validation_gap_mae_eV"]
    nominated = (
        baseline_mae - candidate_mae >= MIN_GAIN_EV
        and control_mae - candidate_mae >= MIN_GAIN_EV
    )
    summary = {
        "format": "molgap-qm9-gape-lite-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "results": results,
        "candidate_gain_vs_baseline_eV": baseline_mae - candidate_mae,
        "candidate_gain_vs_equal_compute_control_eV": control_mae - candidate_mae,
        "required_gain_eV": MIN_GAIN_EV,
        "pcqm_transfer_nominated": nominated,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    atomic_json(output_root / "metrics.json", summary)
    atomic_json(
        output_root / "completion_manifest.json",
        {
            **summary,
            "artifact_sha256": {
                str(path.relative_to(output_root)): sha256_file(path)
                for path in sorted(output_root.rglob("*"))
                if path.is_file() and path.name != "completion_manifest.json"
            },
        },
    )
    return summary
