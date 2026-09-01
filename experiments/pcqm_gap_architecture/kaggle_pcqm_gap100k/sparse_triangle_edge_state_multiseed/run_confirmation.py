"""Kaggle GPU: paired EdgeState/Sparse-Triangle confirmation at seeds 43/44."""
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


OUT = Path("/kaggle/working/pcqm_gap100k_sparse_triangle_edge_state_r3_multiseed")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
EXPECTED_SOURCE_COMMIT = "76dd6efa76c8236ce80a82a8a43d9f5df426165e"
EXPECTED_PARENT_SOURCE_COMMIT = "ba82461c53243d733474c8930ac1b86d82451c91"
EXPECTED_PARENT_AGGREGATE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_WEDGE_AGGREGATE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
COMPARATOR = "ogb_edge_state_structural_gps9"
CANDIDATE = "ogb_sparse_triangle_edge_state_gps9"
CANDIDATES = (
    "ogb_edge_state_structural_gps9",
    "ogb_sparse_triangle_edge_state_gps9",
)
SEEDS = (43, 44)
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1.0e-6
MAX_EPOCHS = 40
PATIENCE = 8
PARAMETER_BUDGET = 5_200_000
SEARCH_BUDGET_S = 39_600
SEED42_REFERENCE = {
    "seed": 42,
    "comparator_validation_gap_mae_eV": 0.13798263211250306,
    "candidate_validation_gap_mae_eV": 0.13790177369117737,
    "candidate_minus_comparator_eV": (
        0.13790177369117737 - 0.13798263211250306
    ),
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
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-sparse-wedge-cache-v1":
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one accepted wedge cache, found {candidates}")
    root, manifest = candidates[0]
    required = {
        "complete": True,
        "parent_cache_source_commit": EXPECTED_PARENT_SOURCE_COMMIT,
        "parent_cache_aggregate_sha256": EXPECTED_PARENT_AGGREGATE_SHA256,
        "aggregate_sha256": EXPECTED_WEDGE_AGGREGATE_SHA256,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "wedge_definition": "directed_nonbacktracking_i_to_j_to_k",
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Wedge cache contract changed for {key}")
    aggregate = hashlib.sha256()
    graph_counts = {"train": 0, "validation": 0}
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Wedge shard hash changed: {shard.get('file')}")
        role = shard.get("role")
        if role not in graph_counts:
            raise RuntimeError(f"Unexpected wedge shard role: {role}")
        graph_counts[role] += int(shard.get("graph_count", 0))
        aggregate.update(
            f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    if graph_counts != {"train": 100_000, "validation": 10_000}:
        raise RuntimeError(f"Wedge graph counts changed: {graph_counts}")
    if aggregate.hexdigest() != EXPECTED_WEDGE_AGGREGATE_SHA256:
        raise RuntimeError("Wedge aggregate hash does not match its shards")
    return root, manifest


def load_graphs(root: Path, manifest: dict) -> dict[str, list]:
    import torch

    graphs = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        payload = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        if len(payload) != int(shard["graph_count"]):
            raise RuntimeError(f"Wedge shard graph count changed: {shard['file']}")
        graphs[shard["role"]].extend(payload)
    if len(graphs["train"]) != 100_000 or len(graphs["validation"]) != 10_000:
        raise RuntimeError("Loaded wedge graph counts changed")
    return graphs


def forward(model, batch, candidate: str):
    arguments = (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    if candidate == CANDIDATE:
        return model(*arguments, batch.wedge_edge_ids)
    return model(*arguments)


def preflight(graphs: dict[str, list]) -> list[dict]:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    device = torch.device("cuda")
    batch = next(
        iter(DataLoader(graphs["train"][:BATCH_SIZE], batch_size=BATCH_SIZE))
    ).to(device)
    results = []
    for candidate in CANDIDATES:
        set_seed(SEEDS[0])
        model = make_pcqm_gap_encoder(candidate).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(f"{candidate} exceeds parameter budget: {parameter_count}")
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
            "finite_prediction": bool(torch.isfinite(prediction).all()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": bool(gradients)
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_s": time.perf_counter() - started,
        }
        if not all(
            row[key] for key in ("finite_prediction", "finite_loss", "finite_gradients")
        ):
            raise RuntimeError(f"Non-finite preflight: {row}")
        results.append(row)
        del model, prediction, loss, gradients
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "format": "molgap-pcqm-gap100k-sparse-triangle-multiseed-preflight-v1",
        "complete": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE,
        "models": results,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(OUT / "preflight.json", payload)
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return results


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


def train_one(
    graphs: dict[str, list],
    candidate: str,
    seed: int,
    task_started: float,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

    if time.perf_counter() - task_started >= SEARCH_BUDGET_S:
        raise TimeoutError("Confirmation budget exhausted before the next paired run")
    set_seed(seed)
    run_dir = OUT / "results" / f"seed{seed}" / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("candidate") != candidate or checkpoint.get("seed") != seed:
            raise RuntimeError("Checkpoint identity changed")
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
    for epoch in range(start_epoch, MAX_EPOCHS):
        if time.perf_counter() - task_started >= SEARCH_BUDGET_S:
            raise TimeoutError("Confirmation budget exhausted during training")
        model.train()
        started = time.perf_counter()
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
        elapsed = time.perf_counter() - started
        improved = validation["mae_eV"] < best_mae
        if improved:
            best_mae = validation["mae_eV"]
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                best_model_path,
                {
                    "candidate": candidate,
                    "seed": seed,
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
                "seed": seed,
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
            f"seed{seed}/{candidate} ep{epoch:02d} "
            f"train={row['train_mae_eV']:.6f} "
            f"val={row['validation_mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break

    best = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation = evaluate(
        model, validation_loader, mean_tensor, std_tensor, device, candidate
    )
    payload_path = run_dir / "validation_payload.pt"
    payload = {
        "candidate": candidate,
        "seed": seed,
        "row_index": validation["row_index"],
        "target_eV": validation["target"],
        "prediction_eV": validation["prediction"],
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_torch_save(payload_path, payload)
    metrics = {
        "format": "molgap-pcqm-gap100k-sparse-triangle-paired-run-v1",
        "complete": True,
        "candidate": candidate,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "seed": seed,
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
        "training_elapsed_s": sum(float(row["elapsed_s"]) for row in trace),
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


def paired_summary(runs: list[dict]) -> tuple[list[dict], bool]:
    by_key = {(row["seed"], row["candidate"]): row for row in runs}
    pairs = [dict(SEED42_REFERENCE)]
    for seed in SEEDS:
        comparator = by_key[(seed, COMPARATOR)]["validation_gap_mae_eV"]
        candidate = by_key[(seed, CANDIDATE)]["validation_gap_mae_eV"]
        pairs.append(
            {
                "seed": seed,
                "comparator_validation_gap_mae_eV": comparator,
                "candidate_validation_gap_mae_eV": candidate,
                "candidate_minus_comparator_eV": candidate - comparator,
            }
        )
    mean_comparator = sum(
        row["comparator_validation_gap_mae_eV"] for row in pairs
    ) / len(pairs)
    mean_candidate = sum(
        row["candidate_validation_gap_mae_eV"] for row in pairs
    ) / len(pairs)
    all_improve = all(row["candidate_minus_comparator_eV"] < 0 for row in pairs)
    passed = all_improve and mean_candidate < mean_comparator
    pairs.append(
        {
            "seed": "mean",
            "comparator_validation_gap_mae_eV": mean_comparator,
            "candidate_validation_gap_mae_eV": mean_candidate,
            "candidate_minus_comparator_eV": mean_candidate - mean_comparator,
        }
    )
    return pairs, passed


def main() -> None:
    task_started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    completed_runs = []
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        source_marker = find_one("PCQM_GAP100K_SOURCE_COMMIT.txt")
        source_commit = source_marker.read_text(encoding="utf-8").strip()
        if source_commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError(f"Source commit changed: {source_commit}")
        cache_root, cache_manifest = cache_root_and_manifest()
        graphs = load_graphs(cache_root, cache_manifest)
        preflight_result = preflight(graphs)
        for seed in SEEDS:
            for candidate in CANDIDATES:
                result = train_one(graphs, candidate, seed, task_started)
                completed_runs.append(result)
                atomic_json(
                    OUT / "progress.json",
                    {
                        "complete": False,
                        "completed": [
                            {"seed": row["seed"], "candidate": row["candidate"]}
                            for row in completed_runs
                        ],
                        "elapsed_s": time.perf_counter() - task_started,
                    },
                )
        pairs, passed = paired_summary(completed_runs)
        selection = {
            "format": "molgap-pcqm-gap100k-sparse-triangle-multiseed-v1",
            "complete": True,
            "source_commit": source_commit,
            "cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
            "parent_cache_aggregate_sha256": cache_manifest[
                "parent_cache_aggregate_sha256"
            ],
            "seeds": list(SEEDS),
            "candidates": list(CANDIDATES),
            "preflight": preflight_result,
            "runs": completed_runs,
            "paired_comparison": pairs,
            "multiseed_gate_passed": passed,
            "selected_candidate": CANDIDATE if passed else COMPARATOR,
            "search_budget_s": SEARCH_BUDGET_S,
            "elapsed_s": time.perf_counter() - task_started,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "selection.json", selection)
        atomic_json(
            OUT / "progress.json",
            {
                "complete": True,
                "completed": [
                    {"seed": row["seed"], "candidate": row["candidate"]}
                    for row in completed_runs
                ],
                "elapsed_s": selection["elapsed_s"],
            },
        )
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed": [
                    {"seed": row["seed"], "candidate": row["candidate"]}
                    for row in completed_runs
                ],
                "elapsed_s": time.perf_counter() - task_started,
            },
        )
        raise


if __name__ == "__main__":
    main()
