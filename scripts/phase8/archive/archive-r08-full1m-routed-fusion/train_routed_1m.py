"""Train the missing 1M base head and test the full-1M routed-v4 topology."""
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
    import torch as probe_torch

    if not probe_torch.cuda.is_available():
        return
    capability = probe_torch.cuda.get_device_capability(0)
    if capability != (6, 0) or "sm_60" in set(probe_torch.cuda.get_arch_list()):
        return
    if os.environ.get("MOLGAP_TORCH_COMPAT_RESTART") == "1":
        raise RuntimeError("cu126 compatibility install still lacks sm_60")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "--no-deps", "--force-reinstall", "torch==2.7.1",
        "nvidia-cusparselt-cu12==0.6.3",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ])
    os.environ["MOLGAP_TORCH_COMPAT_RESTART"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_pascal_compatible_torch()

runtime_paths = sorted(Path("/kaggle/input").rglob("fusion.py")) if Path("/kaggle/input").exists() else []
if runtime_paths:
    sys.path.insert(0, str(runtime_paths[0].parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fusion import FusionHead


INPUT = Path(os.environ.get("MOLGAP_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("MOLGAP_OUTPUT_ROOT", "/kaggle/working"))
SEED = 42
SPLIT_SEED = 42
THRESHOLD_EV = 4.0
MAX_EPOCHS = int(os.environ.get("MOLGAP_MAX_EPOCHS", "150"))
PATIENCE = int(os.environ.get("MOLGAP_PATIENCE", "25"))
BATCH_SIZE = int(os.environ.get("MOLGAP_BATCH_SIZE", "4096"))
EVAL_BATCH_SIZE = int(os.environ.get("MOLGAP_EVAL_BATCH_SIZE", "8192"))
MAX_SAMPLES = int(os.environ["MOLGAP_MAX_SAMPLES"]) if os.environ.get("MOLGAP_MAX_SAMPLES") else None
REFERENCE_AVG = 0.07880732665459315
REFERENCE_GAP = 0.08978520333766937
TAG = "routed_1m_dualgps_seed42"


def find_one(name: str) -> Path:
    paths = sorted(INPUT.rglob(name))
    if not paths:
        raise FileNotFoundError(f"Missing {name} under {INPUT}")
    if len(paths) > 1:
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if len(hashes) != 1:
            raise RuntimeError(f"Conflicting copies of {name}: {paths}")
    return paths[0]


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metric_block(target: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(("HOMO", "LUMO", "Gap")):
        error = target[:, index] - prediction[:, index]
        denominator = float(np.square(target[:, index] - target[:, index].mean()).sum())
        result[name] = {
            "mae_eV": float(np.abs(error).mean()),
            "r2": float(1.0 - np.square(error).sum() / denominator),
        }
    result["average"] = {
        "mae_eV": float(np.mean([result[name]["mae_eV"] for name in ("HOMO", "LUMO", "Gap")])),
        "r2": float(np.mean([result[name]["r2"] for name in ("HOMO", "LUMO", "Gap")])),
    }
    return result


def make_loader(h2: torch.Tensor, h3: torch.Tensor, labels: torch.Tensor,
                indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(h2[indices], h3[indices], labels[indices]),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(model: nn.Module, h2: torch.Tensor, h3: torch.Tensor,
            labels: torch.Tensor, indices: torch.Tensor,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for batch2, batch3, target in make_loader(
        h2, h3, labels, indices, EVAL_BATCH_SIZE, False,
    ):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(batch2.to(device, non_blocking=True), batch3.to(device, non_blocking=True))
        predictions.append(output.float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def train_base(h7: torch.Tensor, h3: torch.Tensor, labels: torch.Tensor,
               split: dict[str, torch.Tensor], split_hash: str,
               device: torch.device) -> tuple[nn.Module, dict]:
    model = FusionHead("gate", 192, 0.0, dim_2d=192, dim_3d=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.4e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.L1Loss()
    best_path = OUTPUT / f"{TAG}_base_best.pt"
    last_path = OUTPUT / f"{TAG}_base_last.pt"
    log_path = OUTPUT / f"{TAG}_base_train_log.csv"
    progress_path = OUTPUT / f"{TAG}_progress.json"
    start_epoch, best_val, best_epoch, wait, log_rows = 0, float("inf"), -1, 0, []

    resume_paths = sorted(INPUT.rglob(last_path.name))
    if resume_paths:
        state = torch.load(resume_paths[0], map_location=device, weights_only=False)
        if state["tag"] != TAG or state["split_sha256"] != split_hash:
            raise RuntimeError("Resume checkpoint does not match tag/split")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["next_epoch"])
        best_val, best_epoch, wait = state["best_val"], state["best_epoch"], state["wait"]
        log_rows = list(state["log"])
        print(f"resuming epoch {start_epoch}; best={best_val:.6f}@{best_epoch}", flush=True)

    train_loader = make_loader(h7, h3, labels, split["train"], BATCH_SIZE, True)
    val_loader = make_loader(h7, h3, labels, split["val"], EVAL_BATCH_SIZE, False)
    for epoch in range(start_epoch, MAX_EPOCHS):
        started = time.time()
        model.train()
        train_total = train_count = 0
        for batch2, batch3, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(
                    model(batch2.to(device, non_blocking=True), batch3.to(device, non_blocking=True)),
                    target.to(device, non_blocking=True),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item()) * len(target)
            train_count += len(target)

        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for batch2, batch3, target in val_loader:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = criterion(
                        model(batch2.to(device, non_blocking=True), batch3.to(device, non_blocking=True)),
                        target.to(device, non_blocking=True),
                    )
                val_total += float(loss.item()) * len(target)
                val_count += len(target)
        val_mae = val_total / val_count
        scheduler.step(val_mae)
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, wait = val_mae, epoch, 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            wait += 1
        row = {
            "epoch": epoch, "train_mae": train_total / train_count,
            "val_mae": val_mae, "best_val_mae": best_val,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        log_rows.append(row)
        temporary_log = log_path.with_name(f".{log_path.name}.tmp")
        pd.DataFrame(log_rows).to_csv(temporary_log, index=False)
        os.replace(temporary_log, log_path)
        atomic_torch_save({
            "tag": TAG, "split_sha256": split_hash, "next_epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
            "log": log_rows,
        }, last_path)
        atomic_json({
            "complete": False, "tag": TAG, "next_epoch": epoch + 1,
            "best_val_mae_eV": best_val, "best_epoch": best_epoch,
        }, progress_path)
        print(
            f"ep{epoch:03d} train={row['train_mae']:.5f} val={val_mae:.5f} "
            f"best={best_val:.5f}@{best_epoch} {row['seconds']:.1f}s"
            + (" *" if improved else ""), flush=True,
        )
        if wait >= PATIENCE:
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, {
        "best_epoch": int(best_epoch), "best_val_mae_eV": float(best_val),
        "epochs_completed": len(log_rows), "best_checkpoint": best_path.name,
        "last_checkpoint": last_path.name, "train_log": log_path.name,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and str(INPUT).startswith("/kaggle"):
        raise RuntimeError("Kaggle GPU accelerator is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if device.type == "cuda":
        smoke = torch.ones((16, 16), device=device) @ torch.ones((16, 16), device=device)
        if float(smoke[0, 0]) != 16.0:
            raise RuntimeError("CUDA matrix preflight failed")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)

    gps7 = torch.load(find_one("gps_expansion_1m_embeddings.pt"), map_location="cpu", weights_only=False)
    gps9 = torch.load(find_one("gps_expansion_1m_depth9_embeddings.pt"), map_location="cpu", weights_only=False)
    schnet = torch.load(find_one("schnet_expansion_1m_embeddings.pt"), map_location="cpu", weights_only=False)
    source7, source9 = gps7["source_idx"].long(), gps9["source_idx"].long()
    source = schnet["source_idx"].long()
    if not torch.equal(source7, source9) or len(source7) != 1_000_000:
        raise RuntimeError("GPS7/GPS9 source coverage differs")
    if len(source) != 997_445 or not torch.all(source[1:] > source[:-1]):
        raise RuntimeError("Unexpected SchNet source coverage/order")
    positions = torch.searchsorted(source7, source)
    if not torch.equal(source7[positions], source):
        raise RuntimeError("GPS and SchNet source_idx alignment failed")
    h7 = gps7["embeddings"][positions].float().contiguous()
    h9 = gps9["embeddings"][positions].float().contiguous()
    h3 = schnet["embeddings"].float().contiguous()
    table = pd.read_csv(find_one("phase8_expansion_1m.csv"), usecols=["homo", "lumo", "gap"])
    labels = torch.from_numpy(table[["homo", "lumo", "gap"]].to_numpy(np.float32, copy=True))[source]
    del gps7, gps9, schnet, table
    if not all(torch.isfinite(value).all() for value in (h7, h9, h3, labels)):
        raise RuntimeError("Non-finite aligned input")

    if MAX_SAMPLES is not None:
        chosen = np.random.RandomState(SPLIT_SEED).permutation(len(source))[:MAX_SAMPLES]
        chosen = torch.from_numpy(np.sort(chosen)).long()
        h7, h9, h3, labels, source = h7[chosen], h9[chosen], h3[chosen], labels[chosen], source[chosen]
    permutation = np.random.RandomState(SPLIT_SEED).permutation(len(source))
    n_train, n_val = int(0.8 * len(source)), int(0.1 * len(source))
    split = {
        "train": torch.from_numpy(permutation[:n_train]).long(),
        "val": torch.from_numpy(permutation[n_train:n_train + n_val]).long(),
        "test": torch.from_numpy(permutation[n_train + n_val:]).long(),
    }
    split_hash = hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()
    print(f"aligned={len(source):,} split={len(split['train']):,}/{len(split['val']):,}/{len(split['test']):,}", flush=True)

    base, training = train_base(h7, h3, labels, split, split_hash, device)
    dual = FusionHead("gate", 192, 0.0, dim_2d=384, dim_3d=192).to(device)
    dual.load_state_dict(torch.load(
        find_one("gate_2gps_expansion_1m_n997445_best.pt"),
        map_location=device, weights_only=True,
    ), strict=True)
    base_prediction, target = predict(base, h7, h3, labels, split["test"], device)
    dual_prediction, dual_target = predict(
        dual, torch.cat([h7, h9], dim=1), h3, labels, split["test"], device,
    )
    if not np.array_equal(target, dual_target):
        raise RuntimeError("Base and dual test targets differ")
    route = base_prediction[:, 2] < THRESHOLD_EV
    routed_prediction = base_prediction.copy()
    routed_prediction[route] = dual_prediction[route]
    base_metrics = metric_block(target, base_prediction)
    dual_metrics = metric_block(target, dual_prediction)
    routed_metrics = metric_block(target, routed_prediction)

    reference_check = {
        "average_delta_eV": dual_metrics["average"]["mae_eV"] - REFERENCE_AVG,
        "gap_delta_eV": dual_metrics["Gap"]["mae_eV"] - REFERENCE_GAP,
    }
    if MAX_SAMPLES is None and max(abs(value) for value in reference_check.values()) > 2e-4:
        raise RuntimeError(f"Existing dual checkpoint reference mismatch: {reference_check}")
    average_error_base = np.abs(base_prediction - target).mean(axis=1)
    average_error_dual = np.abs(dual_prediction - target).mean(axis=1)
    oracle = np.where((average_error_dual < average_error_base)[:, None], dual_prediction, base_prediction)
    result = {
        "complete": True, "tag": TAG, "n_aligned": int(len(source)),
        "split": {"seed": SPLIT_SEED, "sha256": split_hash, **{name: len(value) for name, value in split.items()}},
        "route": {"threshold_eV": THRESHOLD_EV, "count": int(route.sum()), "fraction": float(route.mean())},
        "training": training, "base_metrics": base_metrics,
        "always_dual_metrics": dual_metrics, "routed_metrics": routed_metrics,
        "oracle_molecule_metrics": metric_block(target, oracle),
        "reference_check": reference_check,
        "delta_routed_minus_always_dual": {
            "average_mae_eV": routed_metrics["average"]["mae_eV"] - dual_metrics["average"]["mae_eV"],
            "gap_mae_eV": routed_metrics["Gap"]["mae_eV"] - dual_metrics["Gap"]["mae_eV"],
        },
    }
    atomic_json(result, OUTPUT / f"{TAG}_metrics.json")
    np.savez_compressed(
        OUTPUT / f"{TAG}_test_predictions.npz", source_idx=source[split["test"]].numpy(),
        target=target, base=base_prediction, dual=dual_prediction, routed=routed_prediction,
    )
    atomic_json({
        "complete": True, "tag": TAG, "n_aligned": int(len(source)),
        "split_sha256": split_hash, "outputs": [
            f"{TAG}_base_best.pt", f"{TAG}_base_last.pt",
            f"{TAG}_base_train_log.csv", f"{TAG}_metrics.json",
            f"{TAG}_test_predictions.npz",
        ],
    }, OUTPUT / f"{TAG}_manifest.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
