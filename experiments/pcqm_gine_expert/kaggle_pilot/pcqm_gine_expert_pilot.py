"""Train a bounded PCQM4Mv2 Gap-only GIN virtual-node expert on Kaggle."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


SEED = 42
TRAIN_SAMPLE_ROWS = 250_000
OFFICIAL_VALID_ROWS = 5_000
GRAPH_SHARD_ROWS = 25_000
EPOCHS = 50
BATCH_SIZE = 512
HIDDEN_CHANNELS = 256
NUM_LAYERS = 5
DROPOUT = 0.10
PATIENCE = 7
BASELINE_MAE_EV = 0.2916897588074544
SCALE_GATE_MAE_EV = 0.20
WORK = Path("/kaggle/working")
CACHE = WORK / "pcqm_graph_cache"


def pip_install(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        check=True,
    )


def install_runtime() -> None:
    """Install a P100-compatible runtime and official OGB molecular utilities."""
    pip_install(
        "--upgrade",
        "--force-reinstall",
        "torch==2.7.1+cu126",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    )
    pip_install("--upgrade", "torch_geometric==2.6.1", "ogb==1.3.6")
    pip_install(
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "-f",
        "https://data.pyg.org/whl/torch-2.7.0+cu126.html",
    )
    pip_install(
        "--upgrade",
        "--force-reinstall",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "rdkit==2024.9.6",
    )


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload, torch) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_input_file(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(matches)}: {matches}")
    return matches[0]


def copy_resume_assets() -> None:
    """Reuse explicitly mounted graph/checkpoint outputs when present."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for root in Path("/kaggle/input").iterdir():
        for candidate in root.rglob("graph_shard_*.pt"):
            target = CACHE / candidate.name
            if not target.exists():
                shutil.copy2(candidate, target)
        for name in (
            "pcqm_graph_progress.json",
            "pcqm_gine_last.pt",
            "pcqm_gine_best.pt",
            "pcqm_gine_train_log.csv",
        ):
            matches = list(root.rglob(name))
            if matches:
                target = WORK / name
                if not target.exists():
                    shutil.copy2(matches[0], target)


def load_official_rows(data_path: Path, valid_path: Path, np, pd):
    # The public mirror stores a ZIP archive under the filename data.csv.
    if zipfile.is_zipfile(data_path):
        with zipfile.ZipFile(data_path) as archive:
            with archive.open("data.csv") as source:
                frame = pd.read_csv(
                    source,
                    usecols=["idx", "smiles", "homolumogap"],
                )
    else:
        frame = pd.read_csv(
            data_path,
            usecols=["idx", "smiles", "homolumogap"],
        )

    if len(frame) != 3_746_620 or not np.array_equal(
        frame["idx"].to_numpy(), np.arange(len(frame))
    ):
        raise RuntimeError("Unexpected PCQM4Mv2 row identity/order")
    if not np.isfinite(
        frame["homolumogap"].iloc[:3_378_606].to_numpy()
    ).all():
        raise RuntimeError("PCQM4Mv2 official train labels contain non-finite values")

    # OGB's official train split is the contiguous prefix. The interleaved
    # validation and test indices after it are never read by this training job.
    train_idx = np.arange(3_378_606, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    selected = np.sort(
        rng.choice(train_idx, size=TRAIN_SAMPLE_ROWS, replace=False)
    )
    train_rows = frame.iloc[selected].copy()
    del frame

    valid = pd.read_csv(valid_path).head(OFFICIAL_VALID_ROWS).copy()
    required = {"idx", "smiles", "gap_true"}
    if not required.issubset(valid):
        raise RuntimeError(f"Invalid fixed validation columns: {list(valid)}")
    if len(valid) != OFFICIAL_VALID_ROWS:
        raise RuntimeError(f"Expected {OFFICIAL_VALID_ROWS} validation rows")
    valid = valid.rename(columns={"gap_true": "homolumogap"})
    valid = valid.loc[:, ["idx", "smiles", "homolumogap"]]
    train_rows["source_split"] = np.int8(0)
    valid["source_split"] = np.int8(2)
    rows = pd.concat([train_rows, valid], ignore_index=True)
    return rows, selected, valid["idx"].to_numpy(dtype=np.int64)


def scaffold_bucket(smiles: str, Chem, MurckoScaffold) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    if not scaffold:
        scaffold = "ACYCLIC:" + Chem.MolToSmiles(mol, isomericSmiles=True)
    digest = hashlib.sha1(scaffold.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def build_graph_shards(rows, torch, Data, smiles2graph, Chem, MurckoScaffold):
    existing = sorted(CACHE.glob("graph_shard_*.pt"))
    progress_path = WORK / "pcqm_graph_progress.json"
    if existing and progress_path.exists():
        mounted_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        declared_shards = mounted_progress.get("shards", [])
        declared_hashes = mounted_progress.get("shard_sha256", {})
        if (
            mounted_progress.get("status") == "complete"
            and int(mounted_progress.get("processed_rows", -1)) == len(rows)
            and declared_shards == [path.name for path in existing]
        ):
            for path in existing:
                expected = declared_hashes.get(path.name)
                if not expected or sha256(path) != expected:
                    raise RuntimeError(
                        f"Mounted graph shard failed SHA256 validation: {path.name}"
                    )
            print(
                f"reusing {len(existing)} validated graph shards for "
                f"{len(rows):,} source rows",
                flush=True,
            )
            return

    start = len(existing) * GRAPH_SHARD_ROWS
    if start > len(rows):
        raise RuntimeError("Mounted graph cache has more rows than this configuration")

    progress = {
        "status": "building",
        "source_rows": int(len(rows)),
        "processed_rows": int(start),
        "shards": [path.name for path in existing],
        "invalid_rows": 0,
        "seed": SEED,
    }
    for shard_start in range(start, len(rows), GRAPH_SHARD_ROWS):
        shard_end = min(shard_start + GRAPH_SHARD_ROWS, len(rows))
        graphs = []
        invalid = []
        for row in rows.iloc[shard_start:shard_end].itertuples(index=False):
            try:
                graph = smiles2graph(row.smiles)
            except (AttributeError, RuntimeError, ValueError):
                graph = None
            if graph is None or int(graph["num_nodes"]) == 0:
                invalid.append(int(row.idx))
                continue
            source_split = int(row.source_split)
            if source_split == 0:
                bucket = scaffold_bucket(row.smiles, Chem, MurckoScaffold)
                if bucket < 0:
                    invalid.append(int(row.idx))
                    continue
                split_code = 1 if bucket == 0 else 0
            else:
                split_code = 2
            graphs.append(
                Data(
                    x=torch.as_tensor(graph["node_feat"], dtype=torch.long),
                    edge_index=torch.as_tensor(
                        graph["edge_index"],
                        dtype=torch.long,
                    ),
                    edge_attr=torch.as_tensor(
                        graph["edge_feat"],
                        dtype=torch.long,
                    ),
                    y=torch.tensor([float(row.homolumogap)], dtype=torch.float32),
                    sample_idx=torch.tensor([int(row.idx)], dtype=torch.long),
                    split_code=torch.tensor([split_code], dtype=torch.int8),
                )
            )
        shard_id = shard_start // GRAPH_SHARD_ROWS
        shard_path = CACHE / f"graph_shard_{shard_id:03d}.pt"
        atomic_torch_save(shard_path, graphs, torch)
        progress["processed_rows"] = int(shard_end)
        progress["shards"].append(shard_path.name)
        progress["invalid_rows"] += len(invalid)
        atomic_json(WORK / "pcqm_graph_progress.json", progress)
        print(
            f"graph shard {shard_id}: source={shard_end - shard_start} "
            f"accepted={len(graphs)} invalid={len(invalid)}",
            flush=True,
        )

    progress["status"] = "complete"
    progress["shard_sha256"] = {
        path.name: sha256(path) for path in sorted(CACHE.glob("graph_shard_*.pt"))
    }
    atomic_json(WORK / "pcqm_graph_progress.json", progress)


def load_graphs(torch):
    graphs = []
    for path in sorted(CACHE.glob("graph_shard_*.pt")):
        graphs.extend(torch.load(path, map_location="cpu", weights_only=False))
    if not graphs:
        raise RuntimeError("No graph shards were created")
    train = [graph for graph in graphs if int(graph.split_code.item()) == 0]
    dev = [graph for graph in graphs if int(graph.split_code.item()) == 1]
    official = [graph for graph in graphs if int(graph.split_code.item()) == 2]
    if not train or not dev or len(official) != OFFICIAL_VALID_ROWS:
        raise RuntimeError(
            f"Invalid split counts: train={len(train)} dev={len(dev)} "
            f"official={len(official)}"
        )
    return train, dev, official


def make_model_classes(torch, nn, MessagePassing, AtomEncoder, BondEncoder,
                       global_add_pool, global_mean_pool):
    class OGBGINConv(MessagePassing):
        def __init__(self, hidden_channels: int):
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
            )
            self.eps = nn.Parameter(torch.zeros(1))
            self.bond_encoder = BondEncoder(hidden_channels)

        def forward(self, x, edge_index, edge_attr):
            edge_embedding = self.bond_encoder(edge_attr)
            aggregated = self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_embedding,
            )
            return self.mlp((1 + self.eps) * x + aggregated)

        def message(self, x_j, edge_attr):
            return torch.relu(x_j + edge_attr)

    class GINVirtualNode(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_encoder = AtomEncoder(HIDDEN_CHANNELS)
            self.convs = nn.ModuleList(
                [OGBGINConv(HIDDEN_CHANNELS) for _ in range(NUM_LAYERS)]
            )
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(HIDDEN_CHANNELS) for _ in range(NUM_LAYERS)]
            )
            self.virtual_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(HIDDEN_CHANNELS, 2 * HIDDEN_CHANNELS),
                        nn.BatchNorm1d(2 * HIDDEN_CHANNELS),
                        nn.ReLU(),
                        nn.Linear(2 * HIDDEN_CHANNELS, HIDDEN_CHANNELS),
                        nn.ReLU(),
                    )
                    for _ in range(NUM_LAYERS - 1)
                ]
            )
            self.head = nn.Sequential(
                nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_CHANNELS, 1),
            )

        def forward(self, batch):
            h = self.atom_encoder(batch.x)
            graph_count = int(batch.num_graphs)
            virtual = h.new_zeros((graph_count, HIDDEN_CHANNELS))
            for layer, (conv, norm) in enumerate(
                zip(self.convs, self.batch_norms)
            ):
                h = conv(h + virtual[batch.batch], batch.edge_index, batch.edge_attr)
                h = norm(h)
                if layer != NUM_LAYERS - 1:
                    h = torch.relu(h)
                h = nn.functional.dropout(
                    h,
                    p=DROPOUT,
                    training=self.training,
                )
                if layer < NUM_LAYERS - 1:
                    pooled = global_add_pool(h, batch.batch) + virtual
                    virtual = virtual + nn.functional.dropout(
                        self.virtual_mlps[layer](pooled),
                        p=DROPOUT,
                        training=self.training,
                    )
            return self.head(global_mean_pool(h, batch.batch)).view(-1)

    return GINVirtualNode


def evaluate(model, loader, device, torch):
    model.eval()
    absolute_error = 0.0
    count = 0
    predictions = []
    labels = []
    sample_indices = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch)
            target = batch.y.view(-1)
            absolute_error += torch.abs(prediction - target).sum().item()
            count += int(target.numel())
            predictions.extend(prediction.float().cpu().tolist())
            labels.extend(target.float().cpu().tolist())
            sample_indices.extend(batch.sample_idx.view(-1).cpu().tolist())
    return absolute_error / count, predictions, labels, sample_indices


def train_model(train_graphs, dev_graphs, official_graphs, torch, np):
    import torch.nn as nn
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_add_pool, global_mean_pool
    from torch_geometric.nn.conv import MessagePassing

    device = torch.device("cuda")
    Model = make_model_classes(
        torch,
        nn,
        MessagePassing,
        AtomEncoder,
        BondEncoder,
        global_add_pool,
        global_mean_pool,
    )
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")
    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_graphs,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    official_loader = DataLoader(
        official_graphs,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    best_path = WORK / "pcqm_gine_best.pt"
    last_path = WORK / "pcqm_gine_last.pt"
    log_path = WORK / "pcqm_gine_train_log.csv"
    history = []
    start_epoch = 0
    best_mae = float("inf")
    best_epoch = -1
    stale_epochs = 0
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint["best_dev_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        print(
            f"resuming epoch {start_epoch}; best={best_mae:.6f}@{best_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, EPOCHS):
        started = time.time()
        model.train()
        train_abs = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                prediction = model(batch)
                target = batch.y.view(-1)
                loss = torch.nn.functional.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_abs += torch.abs(prediction.detach() - target).sum().item()
            train_count += int(target.numel())

        train_mae = train_abs / train_count
        dev_mae, _, _, _ = evaluate(model, dev_loader, device, torch)
        scheduler.step(dev_mae)
        improved = dev_mae < best_mae
        if improved:
            best_mae = dev_mae
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                best_path,
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "hidden_channels": HIDDEN_CHANNELS,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "target": "homolumogap",
                        "feature_schema": "ogb_atom_bond",
                        "virtual_node": True,
                    },
                    "epoch": epoch,
                    "dev_mae_eV": dev_mae,
                    "seed": SEED,
                },
                torch,
            )
        else:
            stale_epochs += 1

        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "train_mae_eV": f"{train_mae:.9f}",
            "dev_mae_eV": f"{dev_mae:.9f}",
            "best_dev_mae_eV": f"{best_mae:.9f}",
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}",
            "elapsed_seconds": f"{elapsed:.3f}",
        }
        history.append(row)
        atomic_csv(log_path, history)
        atomic_torch_save(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_dev_mae_eV": best_mae,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "seed": SEED,
            },
            torch,
        )
        print(
            f"ep{epoch:03d} train={train_mae:.5f} dev={dev_mae:.5f} "
            f"best={best_mae:.5f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= PATIENCE and epoch >= 10:
            print(f"early stop after {stale_epochs} stale epochs", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    official_mae, prediction, label, sample_idx = evaluate(
        model,
        official_loader,
        device,
        torch,
    )
    prediction_path = WORK / "pcqm_official_valid_5k_predictions.csv"
    prediction_rows = [
        {
            "idx": int(idx),
            "gap_true_eV": f"{truth:.9f}",
            "gap_prediction_eV": f"{pred:.9f}",
            "absolute_error_eV": f"{abs(pred - truth):.9f}",
        }
        for idx, truth, pred in zip(sample_idx, label, prediction)
    ]
    atomic_csv(prediction_path, prediction_rows)
    metrics = {
        "experiment": "pcqm_gine_expert_pilot",
        "train_rows": len(train_graphs),
        "scaffold_dev_rows": len(dev_graphs),
        "official_valid_rows": len(official_graphs),
        "best_epoch": int(best["epoch"]),
        "best_scaffold_dev_mae_eV": float(best["dev_mae_eV"]),
        "official_valid_5k_gap_mae_eV": float(official_mae),
        "routed_v4_reference_gap_mae_eV": BASELINE_MAE_EV,
        "delta_vs_routed_v4_eV": float(official_mae - BASELINE_MAE_EV),
        "scale_gate_mae_eV": SCALE_GATE_MAE_EV,
        "scale_gate_pass": bool(official_mae <= SCALE_GATE_MAE_EV),
        "official_test_used": False,
        "sealed_20k_used": False,
    }
    atomic_json(WORK / "pcqm_gine_metrics.json", metrics)
    return metrics


def main() -> None:
    install_runtime()

    import numpy as np
    import pandas as pd
    import torch
    from ogb.utils.mol import smiles2graph
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from torch_geometric.data import Data

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not attach a GPU")
    if "sm_60" not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"Torch runtime lacks P100 sm_60 support: {torch.cuda.get_arch_list()}"
        )
    print(
        f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"CUDA={torch.version.cuda}",
        flush=True,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    copy_resume_assets()
    raw_data = find_input_file("data.csv")
    valid_csv = find_input_file("pcqm4mv2_valid_5k.csv")
    raw_hash = sha256(raw_data)
    rows, selected, official_valid = load_official_rows(
        raw_data,
        valid_csv,
        np,
        pd,
    )
    selection = {
        "raw_data_sha256": raw_hash,
        "official_train_sample_rows": int(len(selected)),
        "official_train_sample_idx_sha256": hashlib.sha256(
            selected.tobytes()
        ).hexdigest(),
        "official_valid_rows": int(len(official_valid)),
        "official_valid_idx_sha256": hashlib.sha256(
            official_valid.tobytes()
        ).hexdigest(),
        "seed": SEED,
    }
    atomic_json(WORK / "pcqm_selection_manifest.json", selection)
    build_graph_shards(
        rows,
        torch,
        Data,
        smiles2graph,
        Chem,
        MurckoScaffold,
    )
    train_graphs, dev_graphs, official_graphs = load_graphs(torch)
    print(
        f"graphs train={len(train_graphs):,} scaffold_dev={len(dev_graphs):,} "
        f"official_valid={len(official_graphs):,}",
        flush=True,
    )
    metrics = train_model(
        train_graphs,
        dev_graphs,
        official_graphs,
        torch,
        np,
    )

    artifact_names = [
        "pcqm_selection_manifest.json",
        "pcqm_graph_progress.json",
        "pcqm_gine_best.pt",
        "pcqm_gine_last.pt",
        "pcqm_gine_train_log.csv",
        "pcqm_gine_metrics.json",
        "pcqm_official_valid_5k_predictions.csv",
    ]
    completion = {
        "status": "complete",
        "metrics": metrics,
        "artifacts": {
            name: {
                "bytes": (WORK / name).stat().st_size,
                "sha256": sha256(WORK / name),
            }
            for name in artifact_names
        },
        "graph_shards": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(CACHE.glob("graph_shard_*.pt"))
        },
    }
    atomic_json(WORK / "completion_manifest.json", completion)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
