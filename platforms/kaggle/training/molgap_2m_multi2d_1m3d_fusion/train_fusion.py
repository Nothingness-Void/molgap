"""Train one bounded frozen-embedding 2M-2D plus 1M-3D fusion control."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def ensure_pascal_compatible_torch() -> None:
    """Replace Kaggle's default torch only when it cannot execute on a P100."""
    import torch as probe_torch

    if not probe_torch.cuda.is_available():
        return
    capability = probe_torch.cuda.get_device_capability(0)
    supported_arches = set(probe_torch.cuda.get_arch_list())
    if capability != (6, 0) or "sm_60" in supported_arches:
        return
    if os.environ.get("MOLGAP_TORCH_COMPAT_RESTART") == "1":
        raise RuntimeError(
            "The cu126 compatibility install still lacks sm_60 support; refusing a restart loop"
        )
    print(
        "Kaggle assigned a P100 but the default torch lacks sm_60; "
        "installing torch 2.7.1+cu126 once.",
        flush=True,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "torch==2.7.1",
            "nvidia-cusparselt-cu12==0.6.3",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )
    os.environ["MOLGAP_TORCH_COMPAT_RESTART"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_pascal_compatible_torch()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from fusion import FusionHead


INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 42
VARIANT = json.loads(Path("variant.json").read_text(encoding="utf-8"))["variant"]
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192


def find_one(name: str) -> Path:
    paths = sorted(INPUT.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {name}; found {paths}")
    return paths[0]


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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


def loader(h2, h3, labels, indices, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(h2[indices], h3[indices], labels[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def main() -> None:
    if VARIANT not in {"coverage2m", "hard20k", "multi2d"}:
        raise ValueError(f"Unknown variant: {VARIANT}")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU accelerator is required")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda")
    print(f"variant={VARIANT} gpu={torch.cuda.get_device_name(0)}", flush=True)
    smoke = torch.ones((16, 16), device=device) @ torch.ones((16, 16), device=device)
    if float(smoke[0, 0].item()) != 16.0:
        raise RuntimeError("CUDA matrix preflight returned an invalid result")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"arches={torch.cuda.get_arch_list()} matrix_preflight=OK",
        flush=True,
    )

    coverage = torch.load(find_one("coverage2m_1m_fp16.pt"), map_location="cpu", weights_only=False)
    hard = torch.load(find_one("hard20k_1m_fp16.pt"), map_location="cpu", weights_only=False)
    expected = torch.arange(1_000_000, dtype=torch.long)
    if not torch.equal(coverage["source_idx"], expected) or not torch.equal(hard["source_idx"], expected):
        raise RuntimeError("The staged 2D prefixes are not aligned to the first 1M rows")
    expert_embeddings = {
        "coverage2m": torch.cat([coverage["gps7"], coverage["gps9"]], dim=1),
        "hard20k": torch.cat([hard["gps7"], hard["gps9"]], dim=1),
    }
    h2 = (
        torch.cat([expert_embeddings["coverage2m"], expert_embeddings["hard20k"]], dim=1)
        if VARIANT == "multi2d"
        else expert_embeddings[VARIANT]
    )
    del coverage, hard, expert_embeddings

    schnet = torch.load(find_one("schnet_expansion_1m_embeddings.pt"), map_location="cpu", weights_only=False)
    h3 = schnet["embeddings"].to(torch.float16)
    source = schnet["source_idx"].long()
    if len(source) != 997_445 or source.min() != 0 or source.max() != 999_999:
        raise RuntimeError("Unexpected 1M SchNet source coverage")
    if not torch.all(source[1:] > source[:-1]):
        raise RuntimeError("SchNet source_idx must be sorted and unique")
    h2 = h2[source]
    table = pd.read_csv(find_one("phase8_expansion_1m.csv"), usecols=["homo", "lumo", "gap"])
    labels = torch.from_numpy(table[["homo", "lumo", "gap"]].to_numpy(np.float32, copy=True))[source]
    if not torch.isfinite(h2).all() or not torch.isfinite(h3).all() or not torch.isfinite(labels).all():
        raise RuntimeError("Non-finite aligned fusion input")
    print(f"aligned={len(source):,} dim2d={h2.shape[1]} dim3d={h3.shape[1]}", flush=True)

    permutation = np.random.RandomState(SEED).permutation(len(source))
    n_train, n_val = int(0.8 * len(source)), int(0.1 * len(source))
    split = {
        "train": torch.from_numpy(permutation[:n_train]).long(),
        "val": torch.from_numpy(permutation[n_train:n_train + n_val]).long(),
        "test": torch.from_numpy(permutation[n_train + n_val:]).long(),
    }
    split_hash = hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()

    model = FusionHead(fusion_type="gate", hidden=192, dropout=0.0, dim_2d=h2.shape[1], dim_3d=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.L1Loss()
    tag = f"{VARIANT}_1m3d_seed{SEED}"
    best_path, last_path = OUTPUT / f"{tag}_best.pt", OUTPUT / f"{tag}_last.pt"
    metrics_path, log_path = OUTPUT / f"{tag}_metrics.json", OUTPUT / f"{tag}_train_log.csv"
    start_epoch, best_val, best_epoch, wait, log_rows = 0, float("inf"), -1, 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["tag"] != tag or state["split_sha256"] != split_hash:
            raise RuntimeError("Resume checkpoint does not match this variant/split")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["next_epoch"]
        best_val, best_epoch, wait, log_rows = state["best_val"], state["best_epoch"], state["wait"], state["log"]

    train_loader = loader(h2, h3, labels, split["train"], BATCH_SIZE, True)
    val_loader = loader(h2, h3, labels, split["val"], EVAL_BATCH_SIZE, False)
    for epoch in range(start_epoch, MAX_EPOCHS):
        started = time.time()
        model.train()
        total = count = 0
        for batch2, batch3, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(batch2.to(device), batch3.to(device)), target.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(target)
            count += len(target)
        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for batch2, batch3, target in val_loader:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(batch2.to(device), batch3.to(device)), target.to(device))
                val_total += loss.item() * len(target)
                val_count += len(target)
        val_mae = val_total / val_count
        scheduler.step(val_mae)
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            wait += 1
        row = {"epoch": epoch, "train_mae": total / count, "val_mae": val_mae, "best_val_mae": best_val, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        log_rows.append(row)
        temporary_log = log_path.with_name(f".{log_path.name}.tmp")
        pd.DataFrame(log_rows).to_csv(temporary_log, index=False)
        os.replace(temporary_log, log_path)
        atomic_torch_save({"tag": tag, "split_sha256": split_hash, "next_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_val": best_val, "best_epoch": best_epoch, "wait": wait, "log": log_rows}, last_path)
        atomic_json({"complete": False, "tag": tag, "best_val_mae_eV": best_val, "best_epoch": best_epoch, "log": log_rows}, metrics_path)
        print(f"ep{epoch:03d} train={row['train_mae']:.5f} val={val_mae:.5f} best={best_val:.5f}@{best_epoch} {row['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if wait >= PATIENCE:
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch2, batch3, target in loader(h2, h3, labels, split["test"], EVAL_BATCH_SIZE, False):
            with torch.amp.autocast("cuda"):
                prediction = model(batch2.to(device), batch3.to(device))
            predictions.append(prediction.float().cpu().numpy())
            targets.append(target.numpy())
    result = {
        "complete": True,
        "tag": tag,
        "variant": VARIANT,
        "n_aligned": len(source),
        "split": {"seed": SEED, "sha256": split_hash, **{name: len(value) for name, value in split.items()}},
        "embedding_dims": {"2d": int(h2.shape[1]), "3d": int(h3.shape[1])},
        "best_epoch": best_epoch,
        "best_val_mae_eV": best_val,
        "test_metrics": metrics(np.concatenate(predictions), np.concatenate(targets)),
        "reference_1m_fusion": {"average_mae_eV": 0.07880732665459315, "gap_mae_eV": 0.08978520333766937},
        "log": log_rows,
    }
    result["delta_vs_reference_1m_fusion"] = {
        "average_mae_eV": result["test_metrics"]["average"]["mae_eV"] - result["reference_1m_fusion"]["average_mae_eV"],
        "gap_mae_eV": result["test_metrics"]["Gap"]["mae_eV"] - result["reference_1m_fusion"]["gap_mae_eV"],
    }
    atomic_json(result, metrics_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
