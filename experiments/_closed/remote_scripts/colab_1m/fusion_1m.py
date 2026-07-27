"""Drive-backed 1M dual-GPS plus SchNet late-fusion runner for Colab.

Run this from the already-mounted, GPU-enabled Colab runtime after the 1M
SchNet notebook. It resumes 3D embedding shards and fusion checkpoints from
Drive, so a Colab disconnect never requires starting the expensive stages over.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader as TensorDataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.models.schnet import radius_graph

SEED = 42
GRAPH_BATCH_SIZE = 128
FUSION_BATCH_SIZE = 4096
MAX_EPOCHS = 150
PATIENCE = 25
DEVICE = torch.device("cuda")

# Edit only this block for a later expansion. The first run materializes the
# listed graph parts into FULL_GRAPH_NAME; later runs read that one full cache.
RUN_TAG = "expansion_1m"
GRAPH_PART_NAMES = (
    "pyg_3d_graphs_etkdg_expansion_500k.pt",
    "pyg_3d_graphs_etkdg_expansion_1m_topup_full.pt",
)
FULL_GRAPH_NAME = "pyg_3d_graphs_etkdg_expansion_1m.pt"
GPS_EMBEDDING_NAMES = (
    "gps_expansion_1m_embeddings.pt",
    "gps_expansion_1m_depth9_embeddings.pt",
)
SCHNET_CHECKPOINT_NAME = "extend_1m_n997445_best.pt"
SCHNET_EMBEDDING_NAME = "schnet_expansion_1m_embeddings.pt"
EXPECTED_N_GRAPHS = 997_445
EXPECTED_2D_ROWS = 1_000_000

DRIVE = Path("/content/drive/MyDrive/MolGap")
RESULTS_ROOT = DRIVE / "results"
CHECKPOINTS_ROOT = DRIVE / "checkpoints"
RUN_DIR = RESULTS_ROOT / f"molgap_phase8_colab_fusion_{RUN_TAG}"
CHECKPOINT_DIR = CHECKPOINTS_ROOT / f"molgap_phase8_colab_fusion_{RUN_TAG}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def find_one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}; found {hits}")
    return hits[0]


def find_one_any(roots: tuple[Path, ...], name: str) -> Path:
    hits = sorted(path for root in roots for path in root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {roots}; found {hits}")
    return hits[0]


class SchNetWrapper(nn.Module):
    """Exact architecture used by colab_schnet_3d_1m.ipynb."""

    def __init__(self):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=192,
            num_filters=192,
            num_interactions=6,
            num_gaussians=50,
            cutoff=6.0,
        )
        self.charge_proj = nn.Linear(1, 192)
        self.head = nn.Sequential(
            nn.Linear(192, 192), nn.SiLU(), nn.Dropout(0.0),
            nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 3),
        )

    def encode(self, z, pos, batch, charges=None):
        h = self.schnet.embedding(z)
        if charges is not None:
            h = h + self.charge_proj(charges.unsqueeze(-1))
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch, max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        return global_mean_pool(h, batch)


class FusionHead(nn.Module):
    """Standard gate fusion, not MoE or a router experiment."""

    def __init__(self, dim_2d: int = 384, dim_3d: int = 192, hidden: int = 192):
        super().__init__()
        self.proj_2d = nn.Linear(dim_2d, hidden)
        self.proj_3d = nn.Linear(dim_3d, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.0),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 3),
        )

    def forward(self, h2, h3):
        h2, h3 = self.proj_2d(h2), self.proj_3d(h3)
        gate = self.gate(torch.cat([h2, h3], dim=-1))
        return self.head(gate * h2 + (1.0 - gate) * h3)


def graph_source(graphs) -> torch.Tensor:
    return torch.cat([graph.source_idx.view(-1).cpu() for graph in graphs])


def load_or_build_full_graphs() -> tuple[list[Data], torch.Tensor, Path]:
    """Persist graph parts once, then make every later run read one full cache."""
    full_path = RESULTS_ROOT / FULL_GRAPH_NAME
    if full_path.exists():
        graphs = torch.load(full_path, map_location="cpu", weights_only=False)
        print("Using complete 3D graph cache:", full_path)
    else:
        part_paths = [find_one(RESULTS_ROOT, name) for name in GRAPH_PART_NAMES]
        parts = [torch.load(path, map_location="cpu", weights_only=False) for path in part_paths]
        graphs = [graph for part in parts for graph in part]
        source = graph_source(graphs)
        if torch.unique(source).numel() != len(graphs) or not torch.all(source[1:] > source[:-1]):
            raise RuntimeError("3D graph parts are not source-sorted and non-overlapping")
        tmp_path = full_path.with_suffix(".pt.tmp")
        torch.save(graphs, tmp_path)
        tmp_path.replace(full_path)
        print("Created complete 3D graph cache:", full_path)
    source = graph_source(graphs)
    if torch.unique(source).numel() != len(graphs) or not torch.all(source[1:] > source[:-1]):
        raise RuntimeError("Full 3D graph cache is not source-sorted and unique")
    if EXPECTED_N_GRAPHS is not None and len(graphs) != EXPECTED_N_GRAPHS:
        raise RuntimeError(f"Expected {EXPECTED_N_GRAPHS} 3D graphs, got {len(graphs)}")
    return graphs, source, full_path


@torch.no_grad()
def extract_schnet_embeddings(model, graphs, out: Path) -> None:
    if out.exists():
        payload = torch.load(out, map_location="cpu", weights_only=False)
        if payload["embeddings"].shape == (len(graphs), 192):
            print("Using completed 3D embedding cache:", out)
            return

    shard_dir = RUN_DIR / "schnet_embedding_shards"
    shard_dir.mkdir(exist_ok=True)
    shard_graphs = 100_000
    model.eval()
    for start in range(0, len(graphs), shard_graphs):
        stop = min(start + shard_graphs, len(graphs))
        shard_path = shard_dir / f"shard_{start:07d}_{stop:07d}.pt"
        if shard_path.exists():
            print("Reusing embedding shard:", shard_path.name)
            continue
        embeddings, source_idx = [], []
        loader = GraphDataLoader(
            graphs[start:stop], batch_size=GRAPH_BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                embedding = model.encode(batch.z, batch.pos, batch.batch, getattr(batch, "charges", None))
            embeddings.append(embedding.float().cpu())
            source_idx.append(batch.source_idx.view(-1).cpu())
            if batch_index % 250 == 0:
                print(f"  shard {start}:{stop} batch {batch_index}/{len(loader)}", flush=True)
        torch.save({"embeddings": torch.cat(embeddings), "source_idx": torch.cat(source_idx)}, shard_path)
        print(f"Saved embedding shard {start}:{stop}", flush=True)

    parts = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(shard_dir.glob("shard_*.pt"))]
    payload = {
        "embeddings": torch.cat([part["embeddings"] for part in parts]),
        "source_idx": torch.cat([part["source_idx"] for part in parts]),
    }
    if payload["embeddings"].shape != (len(graphs), 192):
        raise RuntimeError(f"Bad assembled 3D embedding shape: {tuple(payload['embeddings'].shape)}")
    torch.save(payload, out)
    print("Saved complete 3D embeddings:", out)


def make_loader(h2, h3, labels, indices, batch_size: int, shuffle: bool):
    return TensorDataLoader(
        TensorDataset(h2[indices], h3[indices], labels[indices]),
        batch_size=batch_size, shuffle=shuffle, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )


def evaluate(model, h2, h3, labels, indices) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction, target = [], []
    with torch.no_grad():
        for b2, b3, by in make_loader(h2, h3, labels, indices, 8192, False):
            with torch.amp.autocast("cuda"):
                output = model(b2.to(DEVICE, non_blocking=True), b3.to(DEVICE, non_blocking=True))
            prediction.append(output.float().cpu().numpy())
            target.append(by.numpy())
    return np.concatenate(prediction), np.concatenate(target)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(("HOMO", "LUMO", "Gap")):
        result[name] = {
            "mae_eV": float(mean_absolute_error(target[:, index], prediction[:, index])),
            "r2": float(r2_score(target[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae_eV": float(np.mean([result[name]["mae_eV"] for name in ("HOMO", "LUMO", "Gap")])),
        "r2": float(np.mean([result[name]["r2"] for name in ("HOMO", "LUMO", "Gap")])),
    }
    return result


def main() -> None:
    assert torch.cuda.is_available(), "Select a Colab GPU runtime before running."
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("GPU:", torch.cuda.get_device_name(0))

    checkpoint_path = find_one(CHECKPOINTS_ROOT, SCHNET_CHECKPOINT_NAME)
    gps7_path = find_one_any((RESULTS_ROOT, CHECKPOINTS_ROOT), GPS_EMBEDDING_NAMES[0])
    gps9_path = find_one_any((RESULTS_ROOT, CHECKPOINTS_ROOT), GPS_EMBEDDING_NAMES[1])
    graph_embeddings_path = RESULTS_ROOT / SCHNET_EMBEDDING_NAME
    graphs, source, full_graph_path = load_or_build_full_graphs()

    schnet = SchNetWrapper().to(DEVICE)
    schnet.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    extract_schnet_embeddings(schnet, graphs, graph_embeddings_path)
    del schnet
    torch.cuda.empty_cache()

    h3_payload = torch.load(graph_embeddings_path, map_location="cpu", weights_only=False)
    h3, h3_source = h3_payload["embeddings"].float(), h3_payload["source_idx"].long()
    assert torch.equal(h3_source, source) and h3.shape == (len(graphs), 192)
    labels = torch.cat([graph.y.float() for graph in graphs])
    del graphs

    gps7 = torch.load(gps7_path, map_location="cpu", weights_only=False)
    gps9 = torch.load(gps9_path, map_location="cpu", weights_only=False)
    source7, source9 = gps7["source_idx"].long(), gps9["source_idx"].long()
    if EXPECTED_2D_ROWS is not None and source7.numel() != EXPECTED_2D_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_2D_ROWS} 2D rows, got {source7.numel()}")
    assert torch.all(source7[1:] > source7[:-1])
    assert torch.equal(source7, source9), "GPS7/GPS9 source_idx differs"
    positions = torch.searchsorted(source7, source)
    assert torch.all(positions < len(source7)) and torch.equal(source7[positions], source)
    h2 = torch.cat([gps7["embeddings"][positions], gps9["embeddings"][positions]], dim=1).float()
    del gps7, gps9

    permutation = np.random.RandomState(SEED).permutation(len(source))
    n_train, n_val = int(0.8 * len(source)), int(0.1 * len(source))
    train_idx = torch.from_numpy(permutation[:n_train]).long()
    val_idx = torch.from_numpy(permutation[n_train:n_train + n_val]).long()
    test_idx = torch.from_numpy(permutation[n_train + n_val:]).long()
    print(f"Aligned fusion split: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")

    tag = f"gate_2gps_{RUN_TAG}_n{len(source)}"
    best_path = CHECKPOINT_DIR / f"{tag}_best.pt"
    last_path = CHECKPOINT_DIR / f"{tag}_last.pt"
    log_path = RUN_DIR / f"{tag}_train_log.csv"
    metrics_path = RUN_DIR / f"{tag}_metrics.json"
    model = FusionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.L1Loss()
    start_epoch, best_val, best_epoch, wait, log_rows = 0, float("inf"), -1, 0, []
    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        assert payload["tag"] == tag
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
        start_epoch, best_val, best_epoch, wait, log_rows = payload["epoch"] + 1, payload["best_val"], payload["best_epoch"], payload["wait"], payload["log"]
        print(f"Resuming fusion epoch {start_epoch}; best={best_val:.5f}@{best_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        t0 = time.time(); model.train(); total = count = 0
        for b2, b3, by in make_loader(h2, h3, labels, train_idx, FUSION_BATCH_SIZE, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(b2.to(DEVICE, non_blocking=True), b3.to(DEVICE, non_blocking=True)), by.to(DEVICE, non_blocking=True))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update()
            total += loss.item() * by.size(0); count += by.size(0)
        model.eval(); total_val = count_val = 0
        with torch.no_grad():
            for b2, b3, by in make_loader(h2, h3, labels, val_idx, 8192, False):
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(b2.to(DEVICE, non_blocking=True), b3.to(DEVICE, non_blocking=True)), by.to(DEVICE, non_blocking=True))
                total_val += loss.item() * by.size(0); count_val += by.size(0)
        val_mae = total_val / count_val; scheduler.step(val_mae); improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        row = {"epoch": epoch, "train_mae": total / count, "val_mae": val_mae, "best_val_mae": best_val, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        log_rows.append(row); pd.DataFrame(log_rows).to_csv(log_path, index=False)
        torch.save({"tag": tag, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_val": best_val, "best_epoch": best_epoch, "wait": wait, "log": log_rows}, last_path)
        print(f"ep{epoch:03d} train={row['train_mae']:.5f} val={val_mae:.5f} best={best_val:.5f}@{best_epoch} {row['seconds']:.0f}s{' *' if improved else ''}")
        if wait >= PATIENCE:
            print("Early stop")
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    prediction, target = evaluate(model, h2, h3, labels, test_idx)
    result = {
        "tag": tag,
        "n_aligned": int(len(source)),
        "split": {"seed": SEED, "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "embedding_dims": {"2d": int(h2.shape[1]), "3d": int(h3.shape[1])},
        "full_graph_cache": str(full_graph_path),
        "best_epoch": int(best_epoch),
        "best_val_mae_eV": float(best_val),
        "test_metrics": metrics(prediction, target),
        "schnet_reference_test": {"average_mae_eV": 0.1143633748, "gap_mae_eV": 0.1363126040},
    }
    result["delta_vs_schnet"] = {
        "average_mae_eV": result["test_metrics"]["average"]["mae_eV"] - result["schnet_reference_test"]["average_mae_eV"],
        "gap_mae_eV": result["test_metrics"]["Gap"]["mae_eV"] - result["schnet_reference_test"]["gap_mae_eV"],
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Best fusion checkpoint:", best_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
