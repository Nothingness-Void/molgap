"""Same-allocation PCQM scratch repeatability calibration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_scratch_repeatability_s42")
EXPECTED_SOURCE_COMMIT = "c2302c7d433dfec73d293cdbd71ee46aa9e4cae5"
EXPECTED_CACHE = "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
EXPECTED_FIXED_ROOT = "4ecd1546c5fc898004809a5b7130bd7963a5eb569120dbee95e65f18ed9cc99e"
EXPECTED_PARAMETERS = 3_665_809
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"
ROLES = ("repeat_a", "repeat_b")
SEED = 42
BATCH_SIZE = 48
EPOCHS = 60
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
WALL_BUDGET_S = 39_600


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
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
    torch.cuda.manual_seed_all(seed)


def gpu_names() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    supported = (
        len(names) == 2 and all("T4" in name for name in names)
    ) or (len(names) == 1 and "P100" in names[0])
    if not supported:
        raise RuntimeError(f"Expected T4x2 or one P100, received {names}")
    return names


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )


def load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fixed_graphs() -> tuple[dict, dict[str, object]]:
    import torch
    from torch.utils.data import ConcatDataset, Subset
    from torch_geometric.data import InMemoryDataset

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            try:
                self.data, self.slices = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except TypeError:
                self.data, self.slices = torch.load(path, map_location="cpu")

    manifests = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
            manifests.append((path, payload))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one fixed PCQM subset manifest: {manifests}")
    manifest_path, manifest = manifests[0]
    if manifest.get("geometry_aggregate_sha256") != EXPECTED_CACHE:
        raise RuntimeError("fixed cache aggregate identity changed")
    if manifest.get("fixed_root_manifest_sha256") != EXPECTED_FIXED_ROOT:
        raise RuntimeError("fixed root identity changed")
    identity = manifest.get("identity", {})
    if identity.get("train_rows") != 100_000 or identity.get("development_rows") != 50_000:
        raise RuntimeError("fixed subset row contract changed")
    if manifest.get("official_validation_role_read") is not False:
        raise RuntimeError("fixed subset touched official validation")
    root = manifest_path.parent
    shards: dict[str, list] = {"train": [], "development": []}
    for shard in manifest["geometry_shards"]:
        path = root / shard["file"]
        if file_sha256(path) != shard["sha256"]:
            raise RuntimeError(f"graph shard hash changed: {shard['file']}")
        payload = PackedGraphDataset(path)
        if len(payload) != int(shard["rows"]):
            raise RuntimeError(f"graph shard count changed: {shard['file']}")
        shards[shard["role"]].append(payload)
    graphs = {
        "train": ConcatDataset(shards["train"]),
        "development": Subset(
            ConcatDataset(shards["development"]), range(10_000)
        ),
    }
    if len(graphs["train"]) != 100_000 or len(graphs["development"]) != 10_000:
        raise RuntimeError("historical 100K/10K replay count changed")
    development_indices = [int(graph.source_idx) for graph in graphs["development"]]
    if development_indices != list(range(100_000, 110_000)):
        raise RuntimeError("development source_idx slice changed")
    return manifest, graphs


def model_args(batch) -> tuple:
    return (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    )


def make_loader(graphs: list, shuffle: bool):
    import torch
    from torch_geometric.loader import DataLoader

    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=True,
    )


def evaluate(model, loader, mean, std, device) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = model(*model_args(batch)) * std + mean
            total += float((prediction - batch.y.view(-1, 1)).abs().sum().item())
            count += int(batch.y.numel())
    return total / count


def worker(role: str) -> None:
    import torch
    import torch.nn.functional as functional
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"{role} sees {torch.cuda.device_count()} GPUs")
    set_seed(SEED)
    manifest, graphs = load_fixed_graphs()
    device = torch.device("cuda:0")
    model = make_pcqm_gap_encoder(CANDIDATE).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    initial_hash = state_sha256(model.state_dict())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"model parameter count changed: {parameters}")
    targets = torch.tensor([float(graph.y.view(-1)[0]) for graph in graphs["train"]])
    mean = float(targets.mean().item())
    std = float(targets.std(unbiased=False).item())
    train_loader = make_loader(graphs["train"], True)
    development_loader = make_loader(graphs["development"], False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1.0e-6
    )
    run_dir = OUT / role
    trace = []
    best = float("inf")
    best40 = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
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
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float(
                (prediction.detach() * std + mean - batch.y.view(-1, 1))
                .abs()
                .sum()
                .item()
            )
            count += int(batch.y.numel())
        scheduler.step()
        development = evaluate(model, development_loader, mean, std, device)
        if not math.isfinite(development):
            raise RuntimeError(f"non-finite development MAE at epoch {epoch}")
        if development < best:
            best = development
            best_epoch = epoch
            atomic_torch_save(run_dir / "best_model.pt", model.state_dict())
        if epoch < 40:
            best40 = min(best40, development)
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": total / count,
            "development_gap_mae_eV": development,
            "elapsed_s": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_json(run_dir / "trace.json", {"epochs": trace})
        atomic_torch_save(
            run_dir / "last_checkpoint.pt",
            {
                "source_commit": EXPECTED_SOURCE_COMMIT,
                "cache_sha256": EXPECTED_CACHE,
                "seed": SEED,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
        print(
            f"{role} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"dev={development:.6f}eV {row['elapsed_s']:.1f}s",
            flush=True,
        )
    payload = {
        "format": "molgap-pcqm-scratch-repeat-worker-s42-v1",
        "complete": True,
        "role": role,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "cache_sha256": manifest["geometry_aggregate_sha256"],
        "development_source_idx_start": 100_000,
        "development_source_idx_stop": 110_000,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "parameter_count": parameters,
        "initial_encoder_sha256": initial_hash,
        "best_through_40_gap_mae_eV": best40,
        "best_validation_gap_mae_eV": best,
        "best_epoch": best_epoch,
        "trace": trace,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(run_dir / "metrics.json", payload)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        names = gpu_names()
        install_dependencies()
        execution_mode = "parallel_t4x2" if len(names) == 2 else "sequential_p100"
        groups = (
            [[(0, ROLES[0]), (1, ROLES[1])]]
            if len(names) == 2
            else [[(0, ROLES[0])], [(0, ROLES[1])]]
        )
        completed_roles = []
        for group in groups:
            processes = []
            for device, role in group:
                environment = os.environ.copy()
                source_root = str(Path(__file__).resolve().parents[1])
                prior_pythonpath = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    source_root
                    if not prior_pythonpath
                    else source_root + os.pathsep + prior_pythonpath
                )
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
                environment["MOLGAP_REPEAT_WORKER"] = "1"
                environment["MOLGAP_REPEAT_ROLE"] = role
                process = subprocess.Popen(
                    [sys.executable, "-m", "molgap.pcqm_repeatability_runner"],
                    env=environment,
                )
                processes.append((role, process))
            while any(process.poll() is None for _, process in processes):
                failed = [
                    role
                    for role, process in processes
                    if process.poll() is not None and process.returncode != 0
                ]
                if failed:
                    for _, process in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise RuntimeError(f"repeatability workers failed: {failed}")
                if time.perf_counter() - started > WALL_BUDGET_S:
                    for _, process in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise TimeoutError("repeatability calibration exceeded 11 hours")
                atomic_json(
                    OUT / "progress.json",
                    {
                        "complete": False,
                        "gpu_names": names,
                        "execution_mode": execution_mode,
                        "completed_roles": completed_roles,
                        "worker_exitcodes": {
                            role: process.poll() for role, process in processes
                        },
                        "elapsed_s": time.perf_counter() - started,
                        "official_validation_role_read": False,
                        "test_dev_role_read": False,
                    },
                )
                time.sleep(10)
            failed = [
                role
                for role, process in processes
                if process.returncode != 0
            ]
            if failed:
                raise RuntimeError(f"repeatability workers failed: {failed}")
            completed_roles.extend(role for role, _ in processes)
        metrics = {
            role: json.loads((OUT / role / "metrics.json").read_text(encoding="utf-8"))
            for role in ROLES
        }
        first, second = (metrics[role] for role in ROLES)
        initial_hashes = {
            str(payload["initial_encoder_sha256"]) for payload in metrics.values()
        }
        if len(initial_hashes) != 1:
            raise RuntimeError("repeat workers did not share one initialization")
        hashes = {
            role: state_sha256(load_torch(OUT / role / "last_checkpoint.pt")["model"])
            for role in ROLES
        }
        first40 = float(first["best_through_40_gap_mae_eV"])
        second40 = float(second["best_through_40_gap_mae_eV"])
        first60 = float(first["best_validation_gap_mae_eV"])
        second60 = float(second["best_validation_gap_mae_eV"])
        selection = {
            "format": "molgap-pcqm-scratch-repeatability-s42-v1",
            "complete": True,
            "gpu_names": names,
            "execution_mode": execution_mode,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "parameter_count": EXPECTED_PARAMETERS,
            "initial_encoder_sha256": initial_hashes.pop(),
            "geometry_cache_aggregate_sha256": EXPECTED_CACHE,
            "repeat_a_best_40_gap_mae_eV": first40,
            "repeat_b_best_40_gap_mae_eV": second40,
            "absolute_best_40_drift_eV": abs(first40 - second40),
            "repeat_a_best_60_gap_mae_eV": first60,
            "repeat_b_best_60_gap_mae_eV": second60,
            "absolute_best_60_drift_eV": abs(first60 - second60),
            "last_model_state_sha256": hashes,
            "last_model_states_bitwise_equal": len(set(hashes.values())) == 1,
            "elapsed_s": time.perf_counter() - started,
            "architecture_ranking_performed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "selection.json", selection)
        atomic_json(OUT / "progress.json", selection)
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    if os.environ.get("MOLGAP_REPEAT_WORKER") == "1":
        worker(os.environ["MOLGAP_REPEAT_ROLE"])
    else:
        main()
