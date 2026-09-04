"""Single-DCU short runtime gate for the frozen PCQM GraphState9 encoder.

This runner intentionally consumes only the accepted official-train-derived
100K/10K cache.  It is a remote runtime gate, not a local model test and not a
scientific architecture search.
"""
from __future__ import annotations

import argparse
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


CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9"
EXPECTED_MODEL_SOURCE_COMMIT = "9068ddb82e6bdf16b841570abbff023b90c07f07"
EXPECTED_CACHE_AGGREGATE = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
EXPECTED_PARAMS = 3_665_809
EXPECTED_TRAIN_GRAPHS = 100_000
EXPECTED_VALIDATION_GRAPHS = 10_000


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch 1.13 does not expose the weights_only keyword.
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def verify_cache(root: Path) -> tuple[dict, dict[str, list]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-pcqm-gap100k-etkdg-geometry-cache-v1",
        "complete": True,
        "aggregate_sha256": EXPECTED_CACHE_AGGREGATE,
        "official_train_rows_read": 3_378_606,
        "train_graphs": EXPECTED_TRAIN_GRAPHS,
        "validation_graphs": EXPECTED_VALIDATION_GRAPHS,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
        "single_conformer": True,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"cache manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    valid_fraction = manifest.get("valid_geometry_fraction")
    if not finite_number(valid_fraction) or float(valid_fraction) < 0.99:
        raise RuntimeError("cache geometry valid fraction is below 0.99")

    failures_path = root / str(manifest["failures_file"])
    if sha256_file(failures_path) != manifest["failures_file_sha256"]:
        raise RuntimeError("cache geometry failure ledger hash changed")

    aggregate = hashlib.sha256()
    graphs: dict[str, list] = {"train": [], "validation": []}
    for shard in manifest["shards"]:
        shard_path = root / shard["file"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise RuntimeError(f"cache shard hash changed: {shard_path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode(
                "ascii"
            )
        )
        payload = load_torch(shard_path)
        if len(payload) != int(shard["graph_count"]):
            raise RuntimeError(f"cache graph count changed: {shard_path.name}")
        if shard["role"] not in graphs:
            raise RuntimeError(f"unexpected cache role: {shard['role']}")
        graphs[shard["role"]].extend(payload)
    if aggregate.hexdigest() != EXPECTED_CACHE_AGGREGATE:
        raise RuntimeError("cache aggregate hash changed")
    if len(graphs["train"]) != EXPECTED_TRAIN_GRAPHS:
        raise RuntimeError("loaded train graph count changed")
    if len(graphs["validation"]) != EXPECTED_VALIDATION_GRAPHS:
        raise RuntimeError("loaded validation graph count changed")
    for role, items in graphs.items():
        for graph in items[:64]:
            if graph.x.shape[1] != 9 or graph.edge_attr.shape[1] != 3:
                raise RuntimeError(f"{role} categorical feature shape changed")
            if tuple(graph.random_walk_pe.shape) != (graph.num_nodes, 16):
                raise RuntimeError(f"{role} RWSE shape changed")
            if tuple(graph.edge_distance.shape) != (graph.edge_index.shape[1], 1):
                raise RuntimeError(f"{role} distance alignment changed")
            if tuple(graph.wedge_angle_cos.shape) != (
                graph.wedge_edge_ids.shape[0],
                1,
            ):
                raise RuntimeError(f"{role} angle alignment changed")
    return manifest, graphs


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
    ).view(-1)


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def accelerator_snapshot() -> dict:
    commands = []
    for name, args in (
        ("rocm-smi", ["--showuse", "--showmemuse"]),
        ("hy-smi", []),
        ("dcmi", ["diag"]),
    ):
        executable = shutil.which(name)
        if executable is None:
            continue
        try:
            completed = subprocess.run(
                [executable, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            commands.append(
                {
                    "command": name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-2000:],
                }
            )
        except Exception as error:
            commands.append({"command": name, "error": str(error)})
    return {"commands": commands}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-source-commit", default=EXPECTED_MODEL_SOURCE_COMMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if args.seed != 42 or args.epochs != 3 or args.batch_size != 48:
        raise ValueError("Kunshan runtime gate is pinned to seed42/3 epochs/batch48")
    if args.model_source_commit != EXPECTED_MODEL_SOURCE_COMMIT:
        raise ValueError("model source commit changed")

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        import torch
        from torch_geometric.loader import DataLoader

        if not torch.cuda.is_available():
            raise RuntimeError("no DCU/CUDA-compatible device visible")
        device = torch.device("cuda:0")
        set_seed(args.seed)
        manifest, graphs = verify_cache(args.cache_root)

        from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder

        model = make_pcqm_gap_encoder(CANDIDATE)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMS:
            raise RuntimeError(
                f"parameter count changed: {parameter_count} != {EXPECTED_PARAMS}"
            )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.6e-4, weight_decay=1.0e-6
        )
        train_loader = DataLoader(
            graphs["train"], batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(
            graphs["validation"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "unknown"
        run_manifest = {
            "format": "molgap-pcqm-kunshan-graphstate-runtime-v1",
            "candidate": CANDIDATE,
            "model_source_commit": args.model_source_commit,
            "cache_aggregate_sha256": manifest["aggregate_sha256"],
            "seed": args.seed,
            "precision": "fp32",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": 1.6e-4,
            "weight_decay": 1.0e-6,
            "target": "gap",
            "parameter_count": parameter_count,
            "device": device_name,
            "device_count": torch.cuda.device_count(),
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(output / "run_manifest.json", run_manifest)

        epoch_rows = []
        best_validation = float("inf")
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            epoch_started = time.perf_counter()
            train_abs_error = 0.0
            train_count = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(model, batch)
                target = batch.y.view(-1).float()
                loss = torch.nn.functional.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite train loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                train_abs_error += float(torch.sum(torch.abs(prediction - target)).item())
                train_count += int(target.numel())
            synchronize(device)
            train_elapsed = time.perf_counter() - epoch_started

            model.eval()
            validation_abs_error = 0.0
            validation_count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    batch = batch.to(device)
                    prediction = forward(model, batch)
                    target = batch.y.view(-1).float()
                    validation_abs_error += float(
                        torch.sum(torch.abs(prediction - target)).item()
                    )
                    validation_count += int(target.numel())
            synchronize(device)
            train_mae = train_abs_error / train_count
            validation_mae = validation_abs_error / validation_count
            if not finite_number(train_mae) or not finite_number(validation_mae):
                raise RuntimeError(f"non-finite metrics at epoch {epoch}")
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            row = {
                "epoch": epoch,
                "train_mae_eV": train_mae,
                "validation_mae_eV": validation_mae,
                "train_graphs": train_count,
                "validation_graphs": validation_count,
                "epoch_s": time.perf_counter() - epoch_started,
                "train_graphs_per_s": train_count / max(train_elapsed, 1e-9),
                "peak_memory_bytes": peak_memory,
            }
            epoch_rows.append(row)
            checkpoint = {
                "format": "molgap-pcqm-kunshan-graphstate-checkpoint-v1",
                "epoch": epoch,
                "candidate": CANDIDATE,
                "model_source_commit": args.model_source_commit,
                "cache_aggregate_sha256": manifest["aggregate_sha256"],
                "seed": args.seed,
                "model_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "best_validation_mae_eV": min(best_validation, validation_mae),
            }
            atomic_torch_save(output / "last_checkpoint.pt", checkpoint)
            if validation_mae < best_validation:
                best_validation = validation_mae
                best_epoch = epoch
                atomic_torch_save(output / "best_model.pt", model.state_dict())
            atomic_json(
                output / "progress.json",
                {
                    **run_manifest,
                    "complete": False,
                    "latest_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_mae_eV": best_validation,
                    "epochs": epoch_rows,
                    "peak_memory_bytes": max(
                        row["peak_memory_bytes"] for row in epoch_rows
                    ),
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            print(
                f"epoch={epoch} train={train_mae:.8f} val={validation_mae:.8f} "
                f"train_graphs_per_s={row['train_graphs_per_s']:.2f} "
                f"peak_memory_bytes={peak_memory}",
                flush=True,
            )

        accelerator = accelerator_snapshot()
        terminal = {
            **run_manifest,
            "complete": True,
            "best_epoch": best_epoch,
            "best_validation_mae_eV": best_validation,
            "epochs": epoch_rows,
            "elapsed_s": time.perf_counter() - started,
            "peak_memory_bytes": max(row["peak_memory_bytes"] for row in epoch_rows),
            "accelerator_snapshot": accelerator,
            "model_inference_executed": True,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(output / "metrics.json", terminal)
        atomic_json(output / "completion_manifest.json", terminal)
        print(json.dumps(terminal, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            output / "failure.json",
            {
                "format": "molgap-pcqm-kunshan-graphstate-runtime-failure-v1",
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
