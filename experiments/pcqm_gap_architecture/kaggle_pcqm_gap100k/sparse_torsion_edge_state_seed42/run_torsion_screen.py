"""Kaggle GPU: one paired distance/angle-versus-torsion seed-42 screen."""
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


OUT = Path("/kaggle/working/pcqm_gap100k_sparse_torsion_edge_state_s42")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
SEARCH_STARTED_MONOTONIC = "MOLGAP_TORSION_SEARCH_STARTED_MONOTONIC"
EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_PARENT_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
COMPARATOR = "ogb_distance_angle_triangle_edge_state_gps9"
CANDIDATE = "ogb_distance_angle_torsion_triangle_edge_state_gps9"
CANDIDATES = (COMPARATOR, CANDIDATE)
SEED = 42
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
MAX_EPOCHS = 40
PATIENCE = 8
PARAMETER_BUDGET = 5_200_000
SEARCH_BUDGET_S = 23_400
EXPECTED_PARAMETER_COUNTS = {
    COMPARATOR: 4_891_057,
    CANDIDATE: 4_902_081,
}
GEOMETRY_CONTRACT = "ETKDGv3+MMFF94s-single-conformer-bottom-fusion"
TORSION_CONTRACT = "sparse-16-fixed-periodic-shared-gated-update"


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


def search_started() -> float:
    value = os.environ.get(SEARCH_STARTED_MONOTONIC)
    if value is None:
        value = f"{time.monotonic():.9f}"
        os.environ[SEARCH_STARTED_MONOTONIC] = value
    return float(value)


def tensor_sha256(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


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
    extracted = Path("/kaggle/working/_molgap_torsion_screen_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_gap_architecture.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def source_commit_for(source_root: Path) -> str:
    marker = source_root.parent / "PCQM_GAP100K_SOURCE_COMMIT.txt"
    if not marker.is_file():
        marker = source_root / "PCQM_GAP100K_SOURCE_COMMIT.txt"
    if not marker.is_file():
        raise RuntimeError(f"Torsion source marker missing beside {source_root}")
    source_commit = marker.read_text(encoding="utf-8").strip()
    if len(source_commit) != 40:
        raise RuntimeError("Torsion source marker is not a full commit hash")
    return source_commit


def find_torsion_cache(expected_source_commit: str) -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-etkdg-torsion-cache-v1":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one torsion cache, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "source_commit": expected_source_commit,
        "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "torsion_definition": "directed_nonbacktracking_i_j_k_l",
        "torsion_edge_id_shape": ["num_torsions", 3],
        "torsion_wedge_id_shape": ["num_torsions", 2],
        "torsion_feature_definition": "[sin(phi), cos(phi), sin(2phi), cos(2phi)]",
        "torsion_feature_dtype": "float32",
        "invalid_geometry_policy": "zero_torsion_features_and_zero_path_mask",
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Torsion cache contract changed for {key}")
    failures_path = root / manifest["failures_file"]
    if sha256_file(failures_path) != manifest["failures_file_sha256"]:
        raise RuntimeError("Torsion failure ledger hash changed")
    aggregate = hashlib.sha256()
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Torsion shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Torsion aggregate hash changed")
    return root, manifest


def load_graphs(root: Path, manifest: dict) -> dict[str, list]:
    import torch

    graphs = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        payload = torch.load(
            root / shard["file"], map_location="cpu", weights_only=False
        )
        if len(payload) != int(shard["graph_count"]):
            raise RuntimeError(f"Torsion graph count changed: {shard['file']}")
        graphs[shard["role"]].extend(payload)
    if len(graphs["train"]) != 100_000 or len(graphs["validation"]) != 10_000:
        raise RuntimeError("Loaded torsion role counts changed")
    for role, items in graphs.items():
        for graph in items[: min(128, len(items))]:
            torsion_count = graph.torsion_edge_ids.shape[0]
            if tuple(graph.torsion_edge_ids.shape) != (torsion_count, 3):
                raise RuntimeError(f"{role} torsion edge shape is invalid")
            if tuple(graph.torsion_wedge_ids.shape) != (torsion_count, 2):
                raise RuntimeError(f"{role} torsion wedge shape is invalid")
            if tuple(graph.torsion_fourier.shape) != (torsion_count, 4):
                raise RuntimeError(f"{role} torsion feature shape is invalid")
            if tuple(graph.torsion_valid.shape) != (torsion_count, 1):
                raise RuntimeError(f"{role} torsion mask shape is invalid")
    return graphs


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward(model, batch, candidate: str):
    base = (
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
    if candidate == COMPARATOR:
        return model(*base)
    if candidate == CANDIDATE:
        return model(
            *base,
            batch.torsion_edge_ids,
            batch.torsion_wedge_ids,
            batch.torsion_fourier,
            batch.torsion_valid,
        )
    raise ValueError(f"Unexpected torsion candidate: {candidate}")


def preflight(
    graphs: dict[str, list], source_commit: str, torsion_cache_sha256: str
) -> list[dict]:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    device = torch.device("cuda")
    batch = next(
        iter(DataLoader(graphs["train"][:BATCH_SIZE], batch_size=BATCH_SIZE))
    ).to(device)
    rows = []
    initial_reference = None
    initial_function_match = None
    for candidate in CANDIDATES:
        set_seed(SEED)
        model = make_pcqm_gap_encoder(candidate).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNTS[candidate]:
            raise RuntimeError(
                f"{candidate} parameter count changed: {parameter_count}"
            )
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
        model.eval()
        with torch.no_grad():
            initial_prediction = forward(model, batch, candidate)
        if candidate == COMPARATOR:
            initial_reference = initial_prediction.detach().cpu()
        else:
            initial_function_match = bool(
                torch.equal(initial_reference, initial_prediction.detach().cpu())
            )
            if not initial_function_match:
                raise RuntimeError("Torsion injection is not zero-initialized")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction = forward(model, batch, candidate)
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
            "initial_function_match": (
                True if candidate == COMPARATOR else initial_function_match
            ),
            "finite_prediction": bool(torch.isfinite(prediction).all()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": bool(gradients)
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_s": time.perf_counter() - started,
        }
        if not all(
            row[key]
            for key in (
                "initial_function_match",
                "finite_prediction",
                "finite_loss",
                "finite_gradients",
            )
        ):
            raise RuntimeError(f"Non-finite or non-identity torsion preflight: {row}")
        rows.append(row)
        del model, prediction, loss, gradients, initial_prediction
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        OUT / "preflight.json",
        {
            "format": "molgap-pcqm-gap100k-sparse-torsion-preflight-v1",
            "complete": True,
            "source_commit": source_commit,
            "torsion_cache_aggregate_sha256": torsion_cache_sha256,
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": BATCH_SIZE,
            "models": rows,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def evaluate(model, loader, target_mean, target_std, device, candidate: str) -> dict:
    import torch

    model.eval()
    predictions = []
    targets = []
    row_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = forward(model, batch, candidate) * target_std + target_mean
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


def expected_contract(candidate: str) -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "precision": "fp32",
        "target": "gap",
        "geometry": GEOMETRY_CONTRACT,
        "torsion": "none" if candidate == COMPARATOR else TORSION_CONTRACT,
    }


def train_one(
    graphs: dict[str, list],
    candidate: str,
    source_commit: str,
    torsion_cache_sha256: str,
    task_started: float,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if time.perf_counter() - task_started >= SEARCH_BUDGET_S:
        raise TimeoutError("Sparse torsion screen budget exhausted before run")
    set_seed(SEED)
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
    if parameter_count != EXPECTED_PARAMETER_COUNTS[candidate]:
        raise RuntimeError(f"{candidate} parameter count changed: {parameter_count}")
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
        if (
            checkpoint.get("candidate") != candidate
            or checkpoint.get("seed") != SEED
            or checkpoint.get("source_commit") != source_commit
            or checkpoint.get("torsion_cache_aggregate_sha256") != torsion_cache_sha256
        ):
            raise RuntimeError("Sparse torsion checkpoint identity changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_generator.set_state(checkpoint["train_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
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
            raise TimeoutError("Sparse torsion screen budget exhausted during training")
        model.train()
        epoch_started = time.perf_counter()
        train_absolute_error = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            normalized_target = (batch.y.view(-1, 1) - mean_tensor) / std_tensor
            normalized_prediction = forward(model, batch, candidate)
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
            model, validation_loader, mean_tensor, std_tensor, device, candidate
        )
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
                    "source_commit": source_commit,
                    "torsion_cache_aggregate_sha256": torsion_cache_sha256,
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
                "train_generator_state": train_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "source_commit": source_commit,
                "torsion_cache_aggregate_sha256": torsion_cache_sha256,
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

    if not best_model_path.is_file():
        raise RuntimeError(f"{candidate} finished without an atomic best model")
    best = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation = evaluate(
        model, validation_loader, mean_tensor, std_tensor, device, candidate
    )
    payload_path = run_dir / "validation_payload.pt"
    atomic_torch_save(
        payload_path,
        {
            "candidate": candidate,
            "seed": SEED,
            "row_index": validation["row_index"],
            "target_eV": validation["target"],
            "prediction_eV": validation["prediction"],
            "source_commit": source_commit,
            "torsion_cache_aggregate_sha256": torsion_cache_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    metrics = {
        "format": "molgap-pcqm-gap100k-sparse-torsion-paired-run-v1",
        "complete": True,
        "candidate": candidate,
        "source_commit": source_commit,
        "torsion_cache_aggregate_sha256": torsion_cache_sha256,
        "seed": SEED,
        "parameter_count": parameter_count,
        "parameter_budget": PARAMETER_BUDGET,
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
        "contract": expected_contract(candidate),
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


def paired_comparison(run: dict) -> tuple[list[dict], bool]:
    comparator = float(run["comparator"]["validation_gap_mae_eV"])
    candidate = float(run["candidate"]["validation_gap_mae_eV"])
    delta = candidate - comparator
    comparison = [
        {
            "seed": SEED,
            "comparator_validation_gap_mae_eV": comparator,
            "candidate_validation_gap_mae_eV": candidate,
            "candidate_minus_comparator_eV": delta,
        }
    ]
    comparison.append(
        {
            "seed": "mean",
            "comparator_validation_gap_mae_eV": comparator,
            "candidate_validation_gap_mae_eV": candidate,
            "candidate_minus_comparator_eV": delta,
        }
    )
    return comparison, delta < 0


def main() -> None:
    task_started = search_started()
    OUT.mkdir(parents=True, exist_ok=True)
    completed_runs = []
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        source_root = source_python_root()
        sys.path.insert(0, str(source_root))
        source_commit = source_commit_for(source_root)
        if source_commit == EXPECTED_GEOMETRY_SOURCE_COMMIT:
            raise RuntimeError("Torsion screen source was not advanced beyond geometry source")
        cache_root, cache_manifest = find_torsion_cache(source_commit)
        graphs = load_graphs(cache_root, cache_manifest)
        preflight_rows = preflight(
            graphs, source_commit, cache_manifest["aggregate_sha256"]
        )
        for candidate in CANDIDATES:
            result = train_one(
                graphs,
                candidate,
                source_commit,
                cache_manifest["aggregate_sha256"],
                task_started,
            )
            completed_runs.append(result)
            atomic_json(
                OUT / "progress.json",
                {
                    "format": "molgap-pcqm-gap100k-sparse-torsion-progress-v1",
                    "complete": False,
                    "source_commit": source_commit,
                    "torsion_cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
                    "completed": [row["candidate"] for row in completed_runs],
                    "elapsed_s": time.perf_counter() - task_started,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
        run_map = {row["candidate"]: row for row in completed_runs}
        comparison, passed = paired_comparison(
            {"comparator": run_map[COMPARATOR], "candidate": run_map[CANDIDATE]}
        )
        selection = {
            "format": "molgap-pcqm-gap100k-sparse-torsion-seed42-v1",
            "complete": True,
            "source_commit": source_commit,
            "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
            "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
            "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
            "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
            "torsion_cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
            "torsion_cache_valid_geometry_rows": 110_000
            - int(cache_manifest["invalid_geometry_rows"]),
            "seed": SEED,
            "candidates": list(CANDIDATES),
            "preflight": preflight_rows,
            "runs": completed_runs,
            "paired_comparison": comparison,
            "scientific_gate_passed": passed,
            "selected_candidate": CANDIDATE if passed else COMPARATOR,
            "search_budget_s": SEARCH_BUDGET_S,
            "elapsed_s": time.perf_counter() - task_started,
            "atomic_checkpoints": True,
            "independently_retrievable_outputs": True,
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
                "format": "molgap-pcqm-gap100k-sparse-torsion-progress-v1",
                "complete": True,
                "source_commit": source_commit,
                "torsion_cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
                "completed": [row["candidate"] for row in completed_runs],
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
                "format": "molgap-pcqm-gap100k-sparse-torsion-failure-v1",
                "type": type(error).__name__,
                "message": str(error),
                "completed": [row["candidate"] for row in completed_runs],
                "elapsed_s": time.perf_counter() - task_started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
