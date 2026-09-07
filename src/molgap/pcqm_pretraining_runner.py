"""Paired, same-database GraphState9 pretraining screens for Kaggle T4x2."""
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


MODE = os.environ.get("MOLGAP_PRETRAIN_MODE", "structure_source_tasks")
if MODE not in {"structure_source_tasks", "etkdg_geometry_denoising"}:
    raise RuntimeError(f"Unsupported pretraining mode: {MODE}")

SEED = 42
OUT = Path(os.environ.get("MOLGAP_PRETRAIN_OUTPUT", f"/kaggle/working/pcqm_{MODE}_s42"))
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"
EXPECTED_PARAMETERS = 3_665_809
EXPECTED_CACHE_SHA256 = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
EXPECTED_GRAPH_SHA256 = "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
EXPECTED_WEDGE_SHA256 = "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
SCRATCH_EPOCHS = 60
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 40
WALL_BUDGET_S = 39_600
ATOM_DIMS = (119, 4, 12, 12, 10, 6, 6, 2, 2)
BOND_TYPES = 5
FRAGMENT_BINS = 64


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
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def verify_host() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, found {names}")
    return names


def install_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "torch-geometric==2.6.1", "ogb==1.3.6"]
    )


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("molgap/pcqm_pretraining_runner.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one pretraining source tree/archive: {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_pretraining_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_pretraining_runner.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def model_args(batch, noisy: bool = False):
    import torch

    distance = batch.edge_distance
    angle = batch.wedge_angle_cos
    if noisy:
        distance = (distance + 0.05 * torch.randn_like(distance)).clamp(0.5, 3.0)
        angle = (angle + 0.08 * torch.randn_like(angle)).clamp(-1.0, 1.0)
    return (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        distance,
        angle,
        batch.geometry_valid,
    )


def index_mean(values, index, groups: int):
    import torch

    output = values.new_zeros((groups, values.shape[-1]))
    output.index_add_(0, index, values)
    counts = values.new_zeros((groups, 1))
    counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


def composition_targets(batch) -> dict:
    import torch
    import torch.nn.functional as functional

    groups = int(batch.num_graphs)
    atom_one_hot = functional.one_hot(batch.x[:, 0].long(), ATOM_DIMS[0]).float()
    atom = index_mean(atom_one_hot, batch.batch, groups)
    source, target = batch.edge_index
    edge_batch = batch.batch[source]
    bond = index_mean(
        functional.one_hot(batch.edge_attr[:, 0].long(), BOND_TYPES).float(),
        edge_batch,
        groups,
    )
    fragment_id = (
        batch.x[source, 0].long() * 131
        + batch.x[target, 0].long() * 17
        + batch.edge_attr[:, 0].long()
    ) % FRAGMENT_BINS
    fragment = index_mean(
        functional.one_hot(fragment_id, FRAGMENT_BINS).float(), edge_batch, groups
    )
    other = batch.x[:, 1:].float()
    divisors = other.new_tensor([max(1, dim - 1) for dim in ATOM_DIMS[1:]])
    other = index_mean(other / divisors, batch.batch, groups)
    rwse = index_mean(batch.random_walk_pe.float(), batch.batch, groups)
    ones = rwse.new_ones((batch.x.shape[0], 1))
    node_count = rwse.new_zeros((groups, 1)).index_add_(0, batch.batch, ones)
    edge_count = rwse.new_zeros((groups, 1)).index_add_(
        0, edge_batch, rwse.new_ones((source.shape[0], 1))
    )
    wedge_batch = batch.batch[batch.edge_index[1, batch.wedge_edge_ids[:, 0]]]
    wedge_count = rwse.new_zeros((groups, 1)).index_add_(
        0, wedge_batch, rwse.new_ones((wedge_batch.shape[0], 1))
    )
    size = torch.cat(
        [(node_count / 64).clamp_max(1), (edge_count / 128).clamp_max(1),
         (wedge_count / 512).clamp_max(1)], dim=-1
    )
    return {"atom": atom, "bond": bond, "fragment": fragment,
            "molecule": torch.cat([other, rwse, size], dim=-1)}


def histogram(values, group, groups: int, low: float, high: float, bins: int):
    import torch
    import torch.nn.functional as functional

    ids = ((values.reshape(-1) - low) / (high - low) * bins).floor().long()
    ids = ids.clamp(0, bins - 1)
    return index_mean(functional.one_hot(ids, bins).float(), group, groups)


def geometry_targets(batch) -> dict:
    import torch

    groups = int(batch.num_graphs)
    source = batch.edge_index[0]
    edge_batch = batch.batch[source]
    centers = batch.edge_index[1, batch.wedge_edge_ids[:, 0]]
    wedge_batch = batch.batch[centers]
    distance = batch.edge_distance.float().reshape(-1, 1)
    angle = batch.wedge_angle_cos.float().reshape(-1, 1)
    distance_hist = histogram(distance, edge_batch, groups, 0.5, 3.0, 32)
    angle_hist = histogram(angle, wedge_batch, groups, -1.0, 1.0, 32)
    d_mean = index_mean(distance, edge_batch, groups)
    a_mean = index_mean(angle, wedge_batch, groups)
    d_second = index_mean(distance.square(), edge_batch, groups)
    a_second = index_mean(angle.square(), wedge_batch, groups)
    moments = torch.cat([
        (d_mean / 3.0).clamp(0, 1),
        (d_second / 9.0).clamp(0, 1),
        (a_mean + 1.0) / 2.0,
        a_second.clamp(0, 1),
    ], dim=-1)
    return {"distance": distance_hist, "angle": angle_hist, "moments": moments}


def make_heads(channels: int, mode: str):
    import torch.nn as nn

    def head(outputs: int):
        return nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 192),
                             nn.SiLU(), nn.Linear(192, outputs))

    if mode == "structure_source_tasks":
        return nn.ModuleDict({"atom": head(ATOM_DIMS[0]), "bond": head(BOND_TYPES),
                              "fragment": head(FRAGMENT_BINS), "molecule": head(27)})
    return nn.ModuleDict({"distance": head(32), "angle": head(32), "moments": head(4)})


def distribution_loss(logits, target):
    import torch.nn.functional as functional

    return -(target * functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def pretraining_loss(model, heads, batch):
    import torch.nn.functional as functional

    noisy = MODE == "etkdg_geometry_denoising"
    embedding = model.encode(*model_args(batch, noisy=noisy))
    if MODE == "structure_source_tasks":
        target = composition_targets(batch)
        return (
            distribution_loss(heads["atom"](embedding), target["atom"])
            + distribution_loss(heads["bond"](embedding), target["bond"])
            + functional.binary_cross_entropy_with_logits(
                heads["fragment"](embedding), (target["fragment"] > 0).float()
            )
            + functional.smooth_l1_loss(heads["molecule"](embedding), target["molecule"])
        )
    target = geometry_targets(batch)
    return (
        distribution_loss(heads["distance"](embedding), target["distance"])
        + distribution_loss(heads["angle"](embedding), target["angle"])
        + functional.smooth_l1_loss(heads["moments"](embedding), target["moments"])
    )


def make_loader(graphs, shuffle: bool, seed: int):
    import torch
    from torch_geometric.loader import DataLoader

    return DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0,
                      pin_memory=True)


def target_stats(graphs):
    import torch

    values = torch.tensor([float(graph.y.view(-1)[0]) for graph in graphs])
    return float(values.mean()), float(values.std(unbiased=False))


def evaluate(model, loader, mean, std, device):
    import torch

    model.eval()
    error = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = model(*model_args(batch)) * std + mean
            error += float((prediction - batch.y.view(-1, 1)).abs().sum())
            count += batch.y.numel()
    return error / count


def train_gap(model, graphs, epochs: int, run_dir: Path, device, stage: str,
              initial_hash: str) -> dict:
    import torch
    import torch.nn.functional as functional

    mean, std = target_stats(graphs["train"])
    train_loader = make_loader(graphs["train"], True, SEED)
    valid_loader = make_loader(graphs["validation"], False, SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    trace = []
    best = float("inf")
    best_epoch = -1
    best40 = float("inf")
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*model_args(batch))
            target = (batch.y.view(-1, 1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Gap loss at {stage} epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float((prediction.detach() * std + mean - batch.y.view(-1, 1)).abs().sum())
            count += batch.y.numel()
        scheduler.step()
        validation = evaluate(model, valid_loader, mean, std, device)
        best = min(best, validation)
        best_epoch = epoch if validation == best else best_epoch
        if epoch < 40:
            best40 = min(best40, validation)
        row = {"epoch": epoch, "train_gap_mae_eV": total / count,
               "validation_gap_mae_eV": validation,
               "elapsed_s": time.perf_counter() - started,
               "graphs_per_s": len(graphs["train"]) / (time.perf_counter() - started),
               "learning_rate": optimizer.param_groups[0]["lr"]}
        trace.append(row)
        atomic_json(run_dir / f"{stage}_trace.json", {"epochs": trace})
        atomic_torch_save(run_dir / f"{stage}_checkpoint.pt", {
            "mode": MODE, "stage": stage, "seed": SEED, "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "initial_encoder_sha256": initial_hash,
            "cache_sha256": EXPECTED_CACHE_SHA256,
        })
        print(f"{MODE}/{stage} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
              f"val={validation:.6f}eV {row['elapsed_s']:.1f}s", flush=True)
    return {"best_validation_gap_mae_eV": best, "best_epoch": best_epoch,
            "best_through_40_gap_mae_eV": best40, "trace": trace}


def train_pretraining(model, graphs, run_dir: Path, device, initial_hash: str) -> dict:
    import torch

    channels = int(model.head[0].in_features)
    heads = make_heads(channels, MODE).to(device)
    loader = make_loader(graphs["train"], True, SEED)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
    )
    trace = []
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        heads.train()
        started = time.perf_counter()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = pretraining_loss(model, heads, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite pretraining loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch.num_graphs)
            count += int(batch.num_graphs)
        scheduler.step()
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "pretraining_loss": total / count,
               "elapsed_s": elapsed, "graphs_per_s": len(graphs["train"]) / elapsed,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        trace.append(row)
        atomic_json(run_dir / "pretraining_trace.json", {"epochs": trace})
        atomic_torch_save(run_dir / "pretraining_checkpoint.pt", {
            "mode": MODE, "stage": "pretraining", "seed": SEED, "epoch": epoch,
            "model": model.state_dict(), "heads": heads.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "initial_encoder_sha256": initial_hash, "cache_sha256": EXPECTED_CACHE_SHA256,
            "gap_labels_read": False,
        })
        print(f"{MODE}/pretrain ep{epoch:02d} loss={row['pretraining_loss']:.6f} "
              f"{elapsed:.1f}s", flush=True)
    final_hash = state_sha256(model)
    del heads, optimizer, scheduler, loader
    gc.collect()
    torch.cuda.empty_cache()
    return {"epochs": PRETRAIN_EPOCHS, "final_encoder_sha256": final_hash, "trace": trace}


def worker(role: str) -> None:
    import torch
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder
    from molgap.pcqm_local_global_runner import find_geometry_cache, load_graphs

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker {role} sees {torch.cuda.device_count()} GPUs")
    set_seed(SEED)
    cache_root, manifest = find_geometry_cache()
    if manifest["aggregate_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("Geometry cache identity changed")
    graphs = load_graphs(cache_root, manifest)
    device = torch.device("cuda:0")
    model = make_pcqm_gap_encoder(CANDIDATE).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"GraphState9 parameter count changed: {parameters}")
    initial_hash = state_sha256(model)
    run_dir = OUT / role
    run_dir.mkdir(parents=True, exist_ok=True)
    if role == "scratch":
        result = train_gap(model, graphs, SCRATCH_EPOCHS, run_dir, device,
                           "scratch_gap", initial_hash)
        payload = {"role": role, "initial_encoder_sha256": initial_hash,
                   "parameter_count": parameters, **result}
    else:
        pretraining = train_pretraining(model, graphs, run_dir, device, initial_hash)
        fine = train_gap(model, graphs, FINETUNE_EPOCHS, run_dir, device,
                         "finetune_gap", initial_hash)
        payload = {"role": role, "initial_encoder_sha256": initial_hash,
                   "parameter_count": parameters, "pretraining": pretraining, **fine}
    payload.update({"mode": MODE, "seed": SEED,
                    "geometry_cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
                    "parent_graph_cache_aggregate_sha256": EXPECTED_GRAPH_SHA256,
                    "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                    "molecular_research_server_accessed": False})
    atomic_json(run_dir / "metrics.json", payload)


def worker_entry() -> None:
    sys.path.insert(0, str(source_python_root()))
    worker(os.environ["MOLGAP_PRETRAIN_WORKER_ROLE"])


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        names = verify_host()
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        processes = []
        for device, role in enumerate(("scratch", "pretrained")):
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["MOLGAP_PRETRAIN_WORKER"] = "1"
            environment["MOLGAP_PRETRAIN_WORKER_ROLE"] = role
            processes.append((role, subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                                     env=environment)))
        while any(process.poll() is None for _, process in processes):
            time.sleep(10)
            if time.perf_counter() - started > WALL_BUDGET_S:
                for _, process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise TimeoutError("Paired pretraining screen exceeded 11-hour budget")
            failed = [role for role, process in processes
                      if process.poll() is not None and process.returncode != 0]
            if failed:
                for _, process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"Pretraining worker failed: {failed}")
            atomic_json(OUT / "progress.json", {
                "complete": False, "mode": MODE, "gpu_names": names,
                "worker_exitcodes": {role: process.poll() for role, process in processes},
                "elapsed_s": time.perf_counter() - started,
                "official_validation_role_read": False, "test_dev_role_read": False,
            })
        scratch = json.loads((OUT / "scratch" / "metrics.json").read_text())
        pretrained = json.loads((OUT / "pretrained" / "metrics.json").read_text())
        if scratch["initial_encoder_sha256"] != pretrained["initial_encoder_sha256"]:
            raise RuntimeError("Paired models did not start from identical encoders")
        delta40 = pretrained["best_validation_gap_mae_eV"] - scratch["best_through_40_gap_mae_eV"]
        delta60 = pretrained["best_validation_gap_mae_eV"] - scratch["best_validation_gap_mae_eV"]
        selection = {
            "format": "molgap-pcqm-graphstate-pretraining-screen-v1",
            "complete": True, "mode": MODE, "seed": SEED, "gpu_names": names,
            "parameter_count": EXPECTED_PARAMETERS,
            "initial_encoder_sha256": scratch["initial_encoder_sha256"],
            "geometry_cache_aggregate_sha256": EXPECTED_CACHE_SHA256,
            "scratch_best_40_gap_mae_eV": scratch["best_through_40_gap_mae_eV"],
            "scratch_best_60_gap_mae_eV": scratch["best_validation_gap_mae_eV"],
            "pretrained_best_40_gap_mae_eV": pretrained["best_validation_gap_mae_eV"],
            "pretrained_minus_scratch40_eV": delta40,
            "pretrained_minus_scratch60_eV": delta60,
            "passes_material_gate": delta40 <= -0.001 and delta60 <= -0.001,
            "elapsed_s": time.perf_counter() - started,
            "official_validation_role_read": False, "test_dev_role_read": False,
            "seed43_44_submitted": False, "full_data_authorized": False,
            "molecular_research_server_accessed": False,
        }
        atomic_json(OUT / "selection.json", selection)
        atomic_json(OUT / "progress.json", {"complete": True, "mode": MODE,
                                            "elapsed_s": selection["elapsed_s"],
                                            "official_validation_role_read": False,
                                            "test_dev_role_read": False})
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(OUT / "failure.json", {"type": type(error).__name__,
                                           "message": str(error), "mode": MODE,
                                           "elapsed_s": time.perf_counter() - started,
                                           "official_validation_role_read": False,
                                           "test_dev_role_read": False})
        raise


if __name__ == "__main__":
    if os.environ.get("MOLGAP_PRETRAIN_WORKER") == "1":
        worker_entry()
    else:
        main()

