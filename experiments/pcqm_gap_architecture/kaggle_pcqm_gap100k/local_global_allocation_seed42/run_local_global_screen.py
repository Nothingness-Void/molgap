"""Kaggle GPU: seed-42 PCQM local/global allocation screen."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_local_global_allocation_seed42")
EXPECTED_MODEL_SOURCE_COMMIT = "c61e147796ee4195b837bd7e5639ab0dfe97b12c"
EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
CANDIDATES = (
    "ogb_distance_angle_triangle_edge_state_gps9",
    "ogb_distance_angle_triangle_edge_state_sparse_gps369",
    "ogb_distance_angle_triangle_edge_state_graph_state9",
)
FULL_GPS = CANDIDATES[0]
EXPECTED_GLOBAL_BLOCKS = {
    CANDIDATES[0]: tuple(range(1, 10)),
    CANDIDATES[1]: (3, 6, 9),
    CANDIDATES[2]: (),
}
FROZEN_COMPARATOR = {
    "candidate": FULL_GPS,
    "seed": 42,
    "validation_gap_mae_eV": 0.1353926807641983,
    "acceptance": "results/geometry_bottom_fusion_multiseed/acceptance.json",
}
SEED = 42
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
MAX_EPOCHS = 40
PATIENCE = 8
PARAMETER_BUDGET = 5_200_000
SEARCH_BUDGET_S = 39_600
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_TOKEN = "T4"
LOADER_WORKERS = 0
DEVICE_ASSIGNMENTS = {
    0: (CANDIDATES[0],),
    1: (CANDIDATES[1], CANDIDATES[2]),
}


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


def tensor_sha256(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def verify_dual_t4_host() -> list[str]:
    """Verify accelerator allocation without initializing CUDA before fork."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(names) != EXPECTED_GPU_COUNT:
        raise RuntimeError(f"Expected two Kaggle T4 GPUs, found {names}")
    if any(EXPECTED_GPU_TOKEN not in name for name in names):
        raise RuntimeError(f"Expected only T4 GPUs, found {names}")
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


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_gap_architecture.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source tree/archive, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_local_global_screen_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_gap_architecture.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def find_geometry_cache() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-etkdg-geometry-cache-v1":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one geometry cache, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_WEDGE_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "single_conformer": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Geometry cache contract changed for {key}")
    if float(manifest.get("valid_geometry_fraction", 0.0)) < 0.99:
        raise RuntimeError("Geometry cache valid fraction is below 0.99")
    failures_path = root / manifest["failures_file"]
    if sha256_file(failures_path) != manifest["failures_file_sha256"]:
        raise RuntimeError("Geometry failure ledger hash changed")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Geometry shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Geometry aggregate hash changed")
    if manifest["aggregate_sha256"] != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError("Geometry cache identity changed")
    return root, manifest


def load_graphs(root: Path, manifest: dict) -> dict[str, list]:
    import torch

    graphs = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        payload = torch.load(
            root / shard["file"], map_location="cpu", weights_only=False
        )
        if len(payload) != int(shard["graph_count"]):
            raise RuntimeError(f"Geometry graph count changed: {shard['file']}")
        graphs[shard["role"]].extend(payload)
    if len(graphs["train"]) != 100_000 or len(graphs["validation"]) != 10_000:
        raise RuntimeError("Loaded geometry role counts changed")
    for role, items in graphs.items():
        for graph in items[: min(64, len(items))]:
            if tuple(graph.pos.shape) != (graph.num_nodes, 3):
                raise RuntimeError(f"{role} position alignment changed")
            if tuple(graph.edge_distance.shape) != (graph.edge_index.shape[1], 1):
                raise RuntimeError(f"{role} distance alignment changed")
            if tuple(graph.wedge_angle_cos.shape) != (
                graph.wedge_edge_ids.shape[0],
                1,
            ):
                raise RuntimeError(f"{role} angle alignment changed")
    return graphs


def set_seed(seed: int, cuda_device: int | None = None) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if cuda_device is not None:
        with torch.cuda.device(cuda_device):
            torch.cuda.manual_seed(seed)


def forward(model, batch):
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
    )


def initialization_preflight() -> list[dict]:
    """Check schedules and matched CPU initialization before CUDA processes."""
    import torch

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    rows = []
    reference_state = None
    for candidate in CANDIDATES:
        set_seed(SEED)
        model = make_pcqm_gap_encoder(candidate)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
        state = model.state_dict()
        shared_parameter_mismatches = []
        if candidate == FULL_GPS:
            reference_state = {
                name: value.detach().clone() for name, value in state.items()
            }
        else:
            for name in sorted(set(reference_state).intersection(state)):
                if not torch.equal(reference_state[name], state[name]):
                    shared_parameter_mismatches.append(name)
            if shared_parameter_mismatches:
                raise RuntimeError(
                    f"Shared initialization changed for {candidate}: "
                    f"{shared_parameter_mismatches}"
                )
        global_blocks = tuple(
            layer
            for layer, block in enumerate(model.convs, start=1)
            if getattr(block, "use_global_attention", True)
        )
        if global_blocks != EXPECTED_GLOBAL_BLOCKS[candidate]:
            raise RuntimeError(
                f"Global schedule changed for {candidate}: {global_blocks}"
            )
        graph_state_present = hasattr(model, "graph_context")
        if graph_state_present != candidate.endswith("graph_state9"):
            raise RuntimeError(f"Graph-state identity changed for {candidate}")
        rows.append(
            {
                "candidate": candidate,
                "parameter_count": parameter_count,
                "global_attention_blocks": list(global_blocks),
                "graph_state_present": graph_state_present,
                "shared_parameter_mismatches": shared_parameter_mismatches,
            }
        )
        del model, state
        gc.collect()
    return rows


def gpu_preflight(
    graphs: dict[str, list],
    candidate: str,
    physical_device_index: int,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Worker {physical_device_index} did not isolate one GPU: "
            f"{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    set_seed(SEED, cuda_device=0)
    batch = next(
        iter(DataLoader(graphs["train"][:BATCH_SIZE], batch_size=BATCH_SIZE))
    ).to(device)
    model = make_pcqm_gap_encoder(candidate).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > PARAMETER_BUDGET:
        raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
    global_blocks = tuple(
        layer
        for layer, block in enumerate(model.convs, start=1)
        if getattr(block, "use_global_attention", True)
    )
    graph_state_present = hasattr(model, "graph_context")
    if global_blocks != EXPECTED_GLOBAL_BLOCKS[candidate]:
        raise RuntimeError(f"Global schedule changed for {candidate}: {global_blocks}")
    if graph_state_present != candidate.endswith("graph_state9"):
        raise RuntimeError(f"Graph-state identity changed for {candidate}")
    gpu_name = torch.cuda.get_device_name(0)
    if EXPECTED_GPU_TOKEN not in gpu_name:
        raise RuntimeError(f"Worker {physical_device_index} is not on T4: {gpu_name}")
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    prediction = forward(model, batch)
    loss = functional.l1_loss(prediction, batch.y.view(-1, 1))
    loss.backward()
    torch.cuda.synchronize(0)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    row = {
        "candidate": candidate,
        "physical_device_index": physical_device_index,
        "visible_device_index": 0,
        "gpu": gpu_name,
        "parameter_count": parameter_count,
        "global_attention_blocks": list(global_blocks),
        "graph_state_present": graph_state_present,
        "finite_prediction": bool(torch.isfinite(prediction).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": bool(gradients)
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "peak_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "elapsed_s": time.perf_counter() - started,
    }
    if not all(
        row[key] for key in ("finite_prediction", "finite_loss", "finite_gradients")
    ):
        raise RuntimeError(f"Non-finite local/global preflight: {row}")
    atomic_json(OUT / "preflight" / f"{candidate}.json", row)
    del model, prediction, loss, gradients, batch
    gc.collect()
    torch.cuda.empty_cache()
    return row


def evaluate(model, loader, target_mean, target_std, device) -> dict:
    import torch

    model.eval()
    predictions = []
    targets = []
    row_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = forward(model, batch) * target_std + target_mean
            predictions.append(prediction.cpu())
            targets.append(batch.y.view(-1, 1).cpu())
            row_indices.append(batch.row_index.view(-1).cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    row_index = torch.cat(row_indices)
    return {
        "mae_eV": float((prediction - target).abs().mean()),
        "prediction": prediction,
        "target": target,
        "row_index": row_index,
    }


def train_one(
    graphs: dict[str, list],
    candidate: str,
    task_started: float,
    physical_device_index: int,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if time.perf_counter() - task_started >= SEARCH_BUDGET_S:
        raise TimeoutError("Local/global screen budget exhausted before next candidate")
    set_seed(SEED, cuda_device=0)
    run_dir = OUT / "results" / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    train_targets = torch.tensor(
        [float(graph.y.view(-1)[0]) for graph in graphs["train"]],
        dtype=torch.float32,
    )
    target_mean = train_targets.mean().item()
    target_std = train_targets.std(unbiased=False).item()
    if not math.isfinite(target_std) or target_std <= 0:
        raise RuntimeError("Invalid target standard deviation")
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        graphs["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=LOADER_WORKERS,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        graphs["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=True,
    )
    device = torch.device("cuda:0")
    model = make_pcqm_gap_encoder(candidate).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > PARAMETER_BUDGET:
        raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1.0e-6
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    best_model_path = run_dir / "best_model.pt"
    start_epoch = 0
    best_epoch = -1
    best_mae = float("inf")
    stale_epochs = 0
    trace = []
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("candidate") != candidate or checkpoint.get("seed") != SEED:
            raise RuntimeError("Local/global checkpoint identity changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_mae"])
        stale_epochs = int(checkpoint["stale_epochs"])
        trace = list(checkpoint["trace"])

    mean_tensor = torch.tensor(target_mean, device=device)
    std_tensor = torch.tensor(target_std, device=device)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS):
        if time.perf_counter() - task_started >= SEARCH_BUDGET_S:
            raise TimeoutError("Local/global screen budget exhausted during training")
        model.train()
        epoch_started = time.perf_counter()
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
        validation = evaluate(model, validation_loader, mean_tensor, std_tensor, device)
        elapsed = time.perf_counter() - epoch_started
        improved = validation["mae_eV"] < best_mae
        if improved:
            best_mae = validation["mae_eV"]
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                best_model_path,
                {
                    "candidate": candidate,
                    "seed": SEED,
                    "model": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_epoch": best_epoch,
                    "best_mae": best_mae,
                    "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
                },
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_mae_eV": train_absolute_error / train_count,
            "validation_mae_eV": validation["mae_eV"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
            "graphs_per_s": len(graphs["train"]) / elapsed,
            "improved": improved,
        }
        trace.append(row)
        atomic_json(run_dir / "trace.json", {"epochs": trace})
        atomic_torch_save(
            checkpoint_path,
            {
                "candidate": candidate,
                "seed": SEED,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_mae": best_mae,
                "stale_epochs": stale_epochs,
                "trace": trace,
                "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
            },
        )
        print(
            f"{candidate} ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"val={row['validation_mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break

    best = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation = evaluate(model, validation_loader, mean_tensor, std_tensor, device)
    payload_path = run_dir / "validation_payload.pt"
    atomic_torch_save(
        payload_path,
        {
            "candidate": candidate,
            "seed": SEED,
            "row_index": validation["row_index"],
            "target_eV": validation["target"],
            "prediction_eV": validation["prediction"],
            "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    metrics = {
        "format": "molgap-pcqm-gap100k-local-global-candidate-v1",
        "complete": True,
        "candidate": candidate,
        "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
        "seed": SEED,
        "parameter_count": parameter_count,
        "parameter_budget": PARAMETER_BUDGET,
        "physical_device_index": physical_device_index,
        "gpu": torch.cuda.get_device_name(0),
        "best_epoch": best_epoch,
        "validation_gap_mae_eV": validation["mae_eV"],
        "validation_rows": int(validation["target"].numel()),
        "validation_row_index_sha256": tensor_sha256(validation["row_index"]),
        "validation_target_sha256": tensor_sha256(validation["target"]),
        "target_mean_eV": target_mean,
        "target_std_eV": target_std,
        "epochs_completed": len(trace),
        "training_elapsed_s": time.perf_counter() - training_started,
        "mean_throughput_graphs_per_s": sum(
            row["graphs_per_s"] for row in trace
        )
        / len(trace),
        "contract": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "precision": "fp32",
            "loader_workers": LOADER_WORKERS,
            "candidate_parallelism": "dual_t4_process_isolation",
            "target": "gap",
            "geometry": "ETKDGv3+MMFF94s-single-conformer-bottom-fusion",
            "global_attention_blocks": list(EXPECTED_GLOBAL_BLOCKS[candidate]),
            "global_mechanism": (
                "gated_graph_state"
                if candidate.endswith("graph_state9")
                else "multihead_attention"
            ),
        },
        "artifacts": {
            "best_model": str(best_model_path.relative_to(OUT)),
            "best_model_sha256": sha256_file(best_model_path),
            "checkpoint": str(checkpoint_path.relative_to(OUT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_payload": str(payload_path.relative_to(OUT)),
            "validation_payload_sha256": sha256_file(payload_path),
            "trace": str((run_dir / "trace.json").relative_to(OUT)),
            "trace_sha256": sha256_file(run_dir / "trace.json"),
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    del model, optimizer, scheduler, train_loader, validation_loader, validation
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def worker_main(
    graphs: dict[str, list],
    candidates: tuple[str, ...],
    physical_device_index: int,
    task_started: float,
) -> None:
    """Run one isolated candidate queue on exactly one physical T4."""
    completed = []
    try:
        import torch

        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"GPU worker {physical_device_index} sees "
                f"{torch.cuda.device_count()} devices"
            )
        for candidate in candidates:
            gpu_preflight(graphs, candidate, physical_device_index)
            result = train_one(
                graphs,
                candidate,
                task_started,
                physical_device_index,
            )
            completed.append(result["candidate"])
            atomic_json(
                OUT / f"worker_gpu{physical_device_index}_progress.json",
                {
                    "complete": len(completed) == len(candidates),
                    "physical_device_index": physical_device_index,
                    "assigned_candidates": list(candidates),
                    "completed_candidates": completed,
                    "elapsed_s": time.perf_counter() - task_started,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
    except Exception as error:
        atomic_json(
            OUT / f"worker_gpu{physical_device_index}_failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "physical_device_index": physical_device_index,
                "assigned_candidates": list(candidates),
                "completed_candidates": completed,
                "elapsed_s": time.perf_counter() - task_started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


def worker_entry() -> None:
    """Fresh-process entry point; CUDA visibility is fixed by the parent."""
    physical_device_index = int(os.environ["MOLGAP_PHYSICAL_DEVICE_INDEX"])
    candidates = tuple(json.loads(os.environ["MOLGAP_WORKER_CANDIDATES"]))
    task_started = float(os.environ["MOLGAP_TASK_STARTED"])
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_python_root()))
    marker = next(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
    source_commit = marker.read_text(encoding="utf-8").strip()
    if source_commit != EXPECTED_MODEL_SOURCE_COMMIT:
        raise RuntimeError(f"Local/global source commit changed: {source_commit}")
    cache_root, cache_manifest = find_geometry_cache()
    graphs = load_graphs(cache_root, cache_manifest)
    worker_main(graphs, candidates, physical_device_index, task_started)


def select(runs: list[dict]) -> tuple[str, bool, list[dict]]:
    by_name = {row["candidate"]: row for row in runs}
    baseline = by_name[FULL_GPS]
    comparisons = []
    for candidate in CANDIDATES[1:]:
        row = by_name[candidate]
        comparisons.append(
            {
                "candidate": candidate,
                "full_gps_validation_gap_mae_eV": baseline[
                    "validation_gap_mae_eV"
                ],
                "candidate_validation_gap_mae_eV": row[
                    "validation_gap_mae_eV"
                ],
                "candidate_minus_full_gps_eV": row[
                    "validation_gap_mae_eV"
                ]
                - baseline["validation_gap_mae_eV"],
                "parameter_delta": row["parameter_count"]
                - baseline["parameter_count"],
                "throughput_ratio": row["mean_throughput_graphs_per_s"]
                / baseline["mean_throughput_graphs_per_s"],
            }
        )
    winner = min(runs, key=lambda row: row["validation_gap_mae_eV"])
    positive = (
        winner["candidate"] != FULL_GPS
        and winner["validation_gap_mae_eV"]
        < baseline["validation_gap_mae_eV"]
    )
    return winner["candidate"], positive, comparisons


def main() -> None:
    task_started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    completed_runs = []
    try:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Dual-T4 worker isolation requires Linux fork")
        gpu_names = verify_dual_t4_host()
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        marker = next(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
        source_commit = marker.read_text(encoding="utf-8").strip()
        if source_commit != EXPECTED_MODEL_SOURCE_COMMIT:
            raise RuntimeError(
                f"Local/global source commit changed: {source_commit}"
            )
        cache_root, cache_manifest = find_geometry_cache()
        initialization_rows = initialization_preflight()
        atomic_json(
            OUT / "initialization_preflight.json",
            {
                "format": "molgap-pcqm-gap100k-local-global-initialization-v1",
                "complete": True,
                "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
                "models": initialization_rows,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        workers = []
        for device_index, candidates in DEVICE_ASSIGNMENTS.items():
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
            environment["MOLGAP_T4_WORKER"] = "1"
            environment["MOLGAP_PHYSICAL_DEVICE_INDEX"] = str(device_index)
            environment["MOLGAP_WORKER_CANDIDATES"] = json.dumps(candidates)
            environment["MOLGAP_TASK_STARTED"] = repr(task_started)
            workers.append(
                (
                    f"molgap-t4-{device_index}",
                    subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve())],
                        env=environment,
                    ),
                )
            )
        while any(process.poll() is None for _, process in workers):
            time.sleep(5)
            completed_names = [
                candidate
                for candidate in CANDIDATES
                if (OUT / "results" / candidate / "metrics.json").is_file()
            ]
            atomic_json(
                OUT / "progress.json",
                {
                    "complete": False,
                    "execution": "dual_t4_candidate_parallel",
                    "gpu_names": gpu_names,
                    "device_assignments": {
                        str(index): list(candidates)
                        for index, candidates in DEVICE_ASSIGNMENTS.items()
                    },
                    "completed_candidates": completed_names,
                    "worker_exitcodes": {
                        name: process.poll() for name, process in workers
                    },
                    "elapsed_s": time.perf_counter() - task_started,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            if time.perf_counter() - task_started > SEARCH_BUDGET_S + 600:
                for _, process in workers:
                    if process.poll() is None:
                        process.terminate()
                raise TimeoutError("Dual-T4 workers exceeded the wall budget")
        failed_workers = [
            name for name, process in workers if process.returncode != 0
        ]
        if failed_workers:
            raise RuntimeError(f"Dual-T4 workers failed: {failed_workers}")
        preflight_rows = []
        initialization_by_candidate = {
            row["candidate"]: row for row in initialization_rows
        }
        for candidate in CANDIDATES:
            gpu_row = json.loads(
                (OUT / "preflight" / f"{candidate}.json").read_text(
                    encoding="utf-8"
                )
            )
            preflight_rows.append(
                {**initialization_by_candidate[candidate], **gpu_row}
            )
            completed_runs.append(
                json.loads(
                    (OUT / "results" / candidate / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        atomic_json(
            OUT / "preflight.json",
            {
                "format": "molgap-pcqm-gap100k-local-global-preflight-v1",
                "complete": True,
                "source_commit": EXPECTED_MODEL_SOURCE_COMMIT,
                "geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
                "execution": "dual_t4_candidate_parallel",
                "gpu_names": gpu_names,
                "batch_size": BATCH_SIZE,
                "models": preflight_rows,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        selected_candidate, positive, comparisons = select(completed_runs)
        selection = {
            "format": "molgap-pcqm-gap100k-local-global-allocation-screen-v1",
            "complete": True,
            "source_commit": source_commit,
            "geometry_cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
            "geometry_valid_fraction": cache_manifest["valid_geometry_fraction"],
            "seed": SEED,
            "candidates": list(CANDIDATES),
            "execution": "dual_t4_candidate_parallel",
            "gpu_names": gpu_names,
            "device_assignments": {
                str(index): list(candidates)
                for index, candidates in DEVICE_ASSIGNMENTS.items()
            },
            "preflight": preflight_rows,
            "runs": completed_runs,
            "frozen_comparator": FROZEN_COMPARATOR,
            "paired_against_fresh_full_gps": comparisons,
            "selected_candidate": selected_candidate,
            "selected_strictly_improves_full_gps": positive,
            "search_budget_s": SEARCH_BUDGET_S,
            "elapsed_s": time.perf_counter() - task_started,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "molecular_research_server_accessed": False,
            "seed43_44_submitted": False,
            "full_data_authorized": False,
        }
        atomic_json(OUT / "selection.json", selection)
        atomic_json(
            OUT / "progress.json",
            {
                "complete": True,
                "completed_candidates": [row["candidate"] for row in completed_runs],
                "elapsed_s": selection["elapsed_s"],
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_candidates": [row["candidate"] for row in completed_runs],
                "elapsed_s": time.perf_counter() - task_started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    if os.environ.get("MOLGAP_T4_WORKER") == "1":
        worker_entry()
    else:
        main()
