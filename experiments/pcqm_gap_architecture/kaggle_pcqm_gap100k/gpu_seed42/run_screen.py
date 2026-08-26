"""Kaggle GPU: matched official-PCQM Gap100K seed-42 architecture screen."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_seed42")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
EXPECTED_SOURCE_COMMIT = "a67724999dbe145b38c2792b86d4e654f5589a20"
CANDIDATES = ("ogb_structural_gps9", "ogb_edge_state_structural_gps9")
SEED = 42
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
MAX_EPOCHS = 40
PATIENCE = 8
PARAMETER_BUDGET = 5_200_000


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


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_gap_architecture.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archive = find_one("src.zip")
    extracted = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, extracted)
    modules = list(extracted.rglob("molgap/pcqm_gap_architecture.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def ensure_pascal_compatible_torch() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a GPU")
    if torch.cuda.get_device_capability(0) != (6, 0):
        return
    if "sm_60" in set(torch.cuda.get_arch_list()):
        return
    if os.environ.get(PASCAL_COMPAT_RESTART) == "1":
        raise RuntimeError("Compatibility install still lacks sm_60")
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
    os.environ[PASCAL_COMPAT_RESTART] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def install_dependencies() -> None:
    import torch

    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    wheel_index = f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ogb==1.3.6",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "torch_scatter",
            "-f",
            wheel_index,
        ]
    )


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cache_root_and_manifest() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("format") == "molgap-pcqm-gap100k-cache-v1":
            candidates.append((path.parent, payload))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one accepted cache manifest, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "official_train_rows_read": 3_378_606,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "failed_graphs": 0,
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": 16,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Cache contract changed for {key}: {manifest.get(key)}")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Cache shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Cache aggregate hash changed")
    return root, manifest


def load_graphs(root: Path, manifest: dict) -> dict[str, list]:
    import torch

    graphs = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        payload = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        if len(payload) != shard["graph_count"]:
            raise RuntimeError(f"Shard graph count changed: {shard['file']}")
        graphs[shard["role"]].extend(payload)
    if len(graphs["train"]) != 100_000 or len(graphs["validation"]) != 10_000:
        raise RuntimeError("Loaded graph counts changed")
    return graphs


def forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )


def preflight(graphs: dict[str, list]) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    device = torch.device("cuda")
    batch = next(
        iter(DataLoader(graphs["train"][:BATCH_SIZE], batch_size=BATCH_SIZE))
    ).to(device)
    rows = []
    for candidate in CANDIDATES:
        set_seed(SEED)
        model = make_pcqm_gap_encoder(candidate).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction = forward(model, batch)
        loss = functional.l1_loss(prediction, batch.y.view(-1, 1))
        loss.backward()
        torch.cuda.synchronize()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        row = {
            "candidate": candidate,
            "parameter_count": parameter_count,
            "finite_prediction": bool(torch.isfinite(prediction).all()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": bool(gradients)
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_s": time.perf_counter() - started,
        }
        if not all(
            row[key]
            for key in ("finite_prediction", "finite_loss", "finite_gradients")
        ):
            raise RuntimeError(f"Non-finite preflight: {row}")
        rows.append(row)
        del model, prediction, loss, gradients
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "format": "molgap-pcqm-gap100k-seed42-preflight-v1",
        "complete": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE,
        "candidates": rows,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(OUT / "preflight.json", result)
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate(model, loader, target_mean, target_std, device):
    import torch

    model.eval()
    absolute_error = 0.0
    count = 0
    predictions = []
    targets = []
    row_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = forward(model, batch) * target_std + target_mean
            target = batch.y.view(-1, 1)
            absolute_error += float((prediction - target).abs().sum())
            count += target.numel()
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            row_indices.append(batch.row_index.view(-1).cpu())
    return {
        "mae_eV": absolute_error / count,
        "prediction": torch.cat(predictions),
        "target": torch.cat(targets),
        "row_index": torch.cat(row_indices),
    }


def train_candidate(candidate: str, graphs: dict[str, list]) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    set_seed(SEED)
    candidate_dir = OUT / "results" / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train_targets = torch.tensor(
        [float(graph.y.view(-1)[0]) for graph in graphs["train"]],
        dtype=torch.float32,
    )
    target_mean = train_targets.mean().item()
    target_std = train_targets.std(unbiased=False).item()
    if not math.isfinite(target_std) or target_std <= 0:
        raise RuntimeError("Invalid target standard deviation")
    train_loader = DataLoader(
        graphs["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        graphs["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    device = torch.device("cuda")
    model = make_pcqm_gap_encoder(candidate).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1.0e-6
    )
    checkpoint_path = candidate_dir / "checkpoint.pt"
    start_epoch = 0
    best_epoch = -1
    best_mae = float("inf")
    stale_epochs = 0
    trace = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_mae_eV"])
        stale_epochs = int(checkpoint["stale_epochs"])
        trace = list(checkpoint["trace"])

    mean_tensor = torch.tensor(target_mean, device=device)
    std_tensor = torch.tensor(target_std, device=device)
    best_model_path = candidate_dir / "best_model.pt"
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        started = time.perf_counter()
        train_absolute_error = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            normalized_target = (batch.y.view(-1, 1) - mean_tensor) / std_tensor
            normalized_prediction = forward(model, batch)
            loss = functional.l1_loss(normalized_prediction, normalized_target)
            loss.backward()
            optimizer.step()
            prediction_eV = normalized_prediction.detach() * std_tensor + mean_tensor
            train_absolute_error += float(
                (prediction_eV - batch.y.view(-1, 1)).abs().sum()
            )
            train_count += batch.y.numel()
        scheduler.step()
        validation = evaluate(
            model, validation_loader, mean_tensor, std_tensor, device
        )
        elapsed = time.perf_counter() - started
        improved = validation["mae_eV"] < best_mae
        if improved:
            best_mae = float(validation["mae_eV"])
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                best_model_path,
                {
                    "candidate": candidate,
                    "model": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "epoch": epoch,
                    "validation_mae_eV": best_mae,
                    "source_commit": EXPECTED_SOURCE_COMMIT,
                },
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_mae_eV": train_absolute_error / train_count,
            "validation_mae_eV": float(validation["mae_eV"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
            "graphs_per_s": len(graphs["train"]) / elapsed,
            "improved": improved,
        }
        trace.append(row)
        atomic_json(candidate_dir / "trace.json", {"epochs": trace})
        atomic_torch_save(
            checkpoint_path,
            {
                "candidate": candidate,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_mae_eV": best_mae,
                "stale_epochs": stale_epochs,
                "trace": trace,
                "source_commit": EXPECTED_SOURCE_COMMIT,
            },
        )
        print(
            f"{candidate} ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"val={row['validation_mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        del validation
        if stale_epochs >= PATIENCE:
            break

    best = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation = evaluate(model, validation_loader, mean_tensor, std_tensor, device)
    payload_path = candidate_dir / "validation_payload.pt"
    atomic_torch_save(
        payload_path,
        {
            "candidate": candidate,
            "row_index": validation["row_index"],
            "target_eV": validation["target"],
            "prediction_eV": validation["prediction"],
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    metrics = {
        "format": "molgap-pcqm-gap100k-candidate-v1",
        "complete": True,
        "candidate": candidate,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "seed": SEED,
        "parameter_count": parameter_count,
        "parameter_budget": PARAMETER_BUDGET,
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": float(validation["mae_eV"]),
        "target_mean_eV": target_mean,
        "target_std_eV": target_std,
        "epochs_completed": len(trace),
        "contract": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "precision": "fp32",
            "target": "homolumogap",
        },
        "artifacts": {
            "best_model": str(best_model_path.relative_to(OUT)),
            "best_model_sha256": sha256_file(best_model_path),
            "checkpoint": str(checkpoint_path.relative_to(OUT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_payload": str(payload_path.relative_to(OUT)),
            "validation_payload_sha256": sha256_file(payload_path),
            "trace": str((candidate_dir / "trace.json").relative_to(OUT)),
            "trace_sha256": sha256_file(candidate_dir / "trace.json"),
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(candidate_dir / "metrics.json", metrics)
    del model, optimizer, scheduler, train_loader, validation_loader, validation
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        source_commit = find_one("PCQM_GAP100K_SOURCE_COMMIT.txt").read_text().strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"Source commit changed: {source_commit}")
        cache_root, cache_manifest = cache_root_and_manifest()
        graphs = load_graphs(cache_root, cache_manifest)
        preflight_result = preflight(graphs)
        results = [train_candidate(candidate, graphs) for candidate in CANDIDATES]
        winner = min(results, key=lambda item: item["validation_gap_mae_eV"])
        challenger = next(
            item for item in results if item["candidate"] == CANDIDATES[1]
        )
        baseline = next(item for item in results if item["candidate"] == CANDIDATES[0])
        summary = {
            "format": "molgap-pcqm-gap100k-seed42-screen-v1",
            "complete": True,
            "source_commit": source_commit,
            "cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
            "preflight": preflight_result,
            "candidates": results,
            "selected_candidate": winner["candidate"],
            "edge_state_strictly_improves": (
                challenger["validation_gap_mae_eV"]
                < baseline["validation_gap_mae_eV"]
            ),
            "elapsed_s": time.perf_counter() - started,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "selection.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
