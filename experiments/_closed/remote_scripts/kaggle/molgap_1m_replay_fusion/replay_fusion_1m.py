"""Train only a replay-weighted 1M dual-GPS FusionHead on Kaggle.

All encoder embeddings are frozen.  This isolates the effect of changing the
old-500K/new-top-up sampling ratio after PCQM4Mv2 localized the 1M regression
to the fusion layer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


SEED = 42
REPLAY_BOUNDARY = 500_000
REPLAY_WEIGHT = 2.0
BATCH_SIZE = 4096
MAX_EPOCHS = 150
PATIENCE = 25


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)


def install_runtime() -> None:
    # Kaggle can attach a P100 (sm_60), unsupported by its stock CUDA 12.8 torch.
    pip_install(
        "--upgrade", "--force-reinstall", "torch==2.7.1+cu126",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    )
    pip_install("--upgrade", "torch_geometric")


def find_input(required: set[str]) -> Path:
    matches = []
    for root, _, names in os.walk("/kaggle/input"):
        if required.issubset(names):
            matches.append(Path(root))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mounted input with {sorted(required)}, found {matches}")
    return matches[0]


def find_runtime() -> Path:
    for root, dirs, _ in os.walk("/kaggle/input"):
        candidate = Path(root) / "molgap"
        if "molgap" in dirs and (candidate / "fusion.py").is_file():
            return Path(root)
    raise FileNotFoundError("MolGap runtime source is not mounted")


def metric(prediction, target):
    import numpy as np

    result = {}
    for index, name in enumerate(("HOMO", "LUMO", "Gap")):
        error = np.abs(prediction[:, index] - target[:, index])
        centered = target[:, index] - target[:, index].mean()
        result[name] = {
            "mae_eV": float(error.mean()),
            "r2": float(1.0 - np.square(prediction[:, index] - target[:, index]).sum() / np.square(centered).sum()),
        }
    result["average"] = {
        "mae_eV": float(sum(result[name]["mae_eV"] for name in ("HOMO", "LUMO", "Gap")) / 3.0),
        "r2": float(sum(result[name]["r2"] for name in ("HOMO", "LUMO", "Gap")) / 3.0),
    }
    return result


def main() -> None:
    install_runtime()

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

    sys.path.insert(0, str(find_runtime()))
    from molgap.fusion import FusionHead

    assert torch.cuda.is_available(), "Kaggle did not attach a GPU"
    assert "sm_60" in torch.cuda.get_arch_list(), "P100-compatible torch was not installed"
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    input_root = find_input({
        "pyg_3d_graphs_etkdg_expansion_1m.pt",
        "gps_expansion_1m_embeddings.pt",
        "gps_expansion_1m_depth9_embeddings.pt",
        "schnet_expansion_1m_embeddings.pt",
    })

    # Data objects provide labels and the 997,445-row source order. The frozen
    # embeddings are aligned against that order before any split is made.
    graphs = torch.load(input_root / "pyg_3d_graphs_etkdg_expansion_1m.pt", map_location="cpu", weights_only=False)
    source = torch.cat([graph.source_idx.view(-1).cpu() for graph in graphs]).long()
    labels = torch.cat([graph.y.view(1, -1).float() for graph in graphs])
    if source.numel() != 997_445 or labels.shape != (997_445, 3):
        raise RuntimeError(f"Unexpected graph payload: source={tuple(source.shape)}, labels={tuple(labels.shape)}")
    if torch.unique(source).numel() != len(source) or not torch.all(source[1:] > source[:-1]):
        raise RuntimeError("3D source_idx must be sorted and unique")

    gps7 = torch.load(input_root / "gps_expansion_1m_embeddings.pt", map_location="cpu", weights_only=False)
    gps9 = torch.load(input_root / "gps_expansion_1m_depth9_embeddings.pt", map_location="cpu", weights_only=False)
    schnet = torch.load(input_root / "schnet_expansion_1m_embeddings.pt", map_location="cpu", weights_only=False)
    for name, payload, expected_dim in (("gps7", gps7, 192), ("gps9", gps9, 192), ("schnet", schnet, 192)):
        if payload["embeddings"].ndim != 2 or payload["embeddings"].shape[1] != expected_dim:
            raise RuntimeError(f"Unexpected {name} embedding shape {tuple(payload['embeddings'].shape)}")
    if not torch.equal(gps7["source_idx"], gps9["source_idx"]):
        raise RuntimeError("GPS7/GPS9 source_idx differs")
    positions = torch.searchsorted(gps7["source_idx"].long(), source)
    if not torch.equal(gps7["source_idx"][positions].long(), source):
        raise RuntimeError("GPS embeddings do not cover every 3D source_idx")
    positions_3d = torch.searchsorted(schnet["source_idx"].long(), source)
    if not torch.equal(schnet["source_idx"][positions_3d].long(), source):
        raise RuntimeError("SchNet embeddings do not cover every 3D source_idx")
    h2 = torch.cat([gps7["embeddings"][positions], gps9["embeddings"][positions]], dim=1).float()
    h3 = schnet["embeddings"][positions_3d].float()
    del graphs, gps7, gps9, schnet

    permutation = np.random.RandomState(SEED).permutation(len(source))
    train_idx = torch.from_numpy(permutation[:int(0.8 * len(source))]).long()
    val_idx = torch.from_numpy(permutation[int(0.8 * len(source)):int(0.9 * len(source))]).long()
    test_idx = torch.from_numpy(permutation[int(0.9 * len(source)):]).long()
    old_train = source[train_idx] < REPLAY_BOUNDARY
    if not old_train.any() or old_train.all():
        raise RuntimeError("Replay boundary did not split the train set")
    train_weights = torch.where(old_train, REPLAY_WEIGHT, 1.0).double()
    replay_report = {
        "source_idx_lt": REPLAY_BOUNDARY,
        "old_train_rows": int(old_train.sum()),
        "new_train_rows": int((~old_train).sum()),
        "old_weight": REPLAY_WEIGHT,
        "expected_old_draw_fraction": float(train_weights[old_train].sum() / train_weights.sum()),
    }

    dataset = TensorDataset(h2, h3, labels)
    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())
    sampler = WeightedRandomSampler(
        train_weights, num_samples=len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=8192, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=8192, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.L1Loss()
    best_val, best_epoch, wait, best_state = float("inf"), -1, 0, None
    log = []
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train(); train_total = train_n = 0
        for batch_h2, batch_h3, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = criterion(model(batch_h2.to(device, non_blocking=True), batch_h3.to(device, non_blocking=True)), batch_y.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer); scaler.update()
            train_total += loss.item() * len(batch_y); train_n += len(batch_y)
        model.eval(); val_total = val_n = 0
        with torch.no_grad():
            for batch_h2, batch_h3, batch_y in val_loader:
                with torch.autocast("cuda"):
                    loss = criterion(model(batch_h2.to(device, non_blocking=True), batch_h3.to(device, non_blocking=True)), batch_y.to(device, non_blocking=True))
                val_total += loss.item() * len(batch_y); val_n += len(batch_y)
        val_mae = val_total / val_n
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val, best_epoch, wait = val_mae, epoch, 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        row = {"epoch": epoch, "train_mae": train_total / train_n, "val_mae": val_mae, "best_val_mae": best_val, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        log.append(row)
        print(f"ep{epoch:03d} train={row['train_mae']:.5f} val={val_mae:.5f} best={best_val:.5f}@{best_epoch} {row['seconds']:.1f}s{' *' if wait == 0 else ''}", flush=True)
        if wait >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No best FusionHead state")
    model.load_state_dict(best_state)
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_h2, batch_h3, batch_y in test_loader:
            with torch.autocast("cuda"):
                predictions.append(model(batch_h2.to(device, non_blocking=True), batch_h3.to(device, non_blocking=True)).float().cpu().numpy())
            targets.append(batch_y.numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    tag = "gate_2gps_expansion_1m_replay2_n997445"
    torch.save(best_state, Path("/kaggle/working") / f"{tag}_best.pt")
    pd.DataFrame(log).to_csv(Path("/kaggle/working") / f"{tag}_train_log.csv", index=False)
    result = {
        "tag": tag,
        "n_aligned": int(len(source)),
        "split": {"seed": SEED, "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "replay_sampling": replay_report,
        "best_epoch": best_epoch,
        "best_val_mae_eV": best_val,
        "test_metrics": metric(prediction, target),
    }
    Path("/kaggle/working/replay_fusion_1m_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
