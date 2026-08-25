"""Kaggle2 GPU stage: preflight and train PairGPS-R2 on fixed QM9."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/pair_gps_2d_r2")
CACHE = Path("/kaggle/working/qm9_r2_cache")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


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

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a GPU")
    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    index = f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "torch_scatter",
            "-f",
            index,
        ]
    )


def stage_inputs() -> Path:
    source = find_one("src/molgap/qm9_screen.py").parents[2]
    acceptance = find_one("acceptance.json")
    source_cache = acceptance.parents[2]
    if CACHE.exists():
        shutil.rmtree(CACHE)
    shutil.copytree(source_cache, CACHE)
    sys.path.insert(0, str(source / "src"))
    return source


def remote_preflight(source: Path) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.qm9_screen import (
        _forward,
        _topology_graph,
        attach_accepted_qm9_rwse,
        fixed_split,
        load_qm9_records,
        make_encoder,
        target_stats,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    records = load_qm9_records(CACHE)
    split = fixed_split(len(records), 30_000, 3_000, 3_000, 42)
    mean, std = target_stats(records, split.train)
    graphs = {
        "train": [
            _topology_graph(records[int(index)], int(index), mean, std)
            for index in split.train[:48]
        ],
        "validation": [],
        "test": [],
    }
    acceptance = attach_accepted_qm9_rwse(
        graphs, cache_dir=CACHE, split=split, walk_length=16
    )
    batch = next(iter(DataLoader(graphs["train"][:48], batch_size=48)))
    model, kind = make_encoder("pair_gps_2d_r2")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 4_740_000:
        raise RuntimeError(f"Parameter budget exceeded: {parameter_count}")
    device = torch.device("cuda")
    model = model.to(device)
    batch = batch.to(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction = _forward(kind, model, batch)
    loss = functional.l1_loss(prediction, batch.y.view(-1, 3))
    loss.backward()
    torch.cuda.synchronize()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    result = {
        "format": "molgap-pairgps-r2-preflight-v1",
        "complete": True,
        "source_commit": (source / "SOURCE_COMMIT.txt").read_text().strip(),
        "candidate": "pair_gps_2d_r2",
        "parameter_count": parameter_count,
        "parameter_budget": 4_740_000,
        "prediction_shape": list(prediction.shape),
        "finite_prediction": bool(torch.isfinite(prediction).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_s": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "split_fingerprint": split.fingerprint,
        "rwse_output_sha256": acceptance["output_sha256"],
        "validation_role_read": False,
        "test_role_read": False,
    }
    if not all(
        result[key]
        for key in ("finite_prediction", "finite_loss", "finite_gradients")
    ):
        raise RuntimeError(f"Non-finite remote preflight: {result}")
    atomic_json(OUT / "preflight.json", result)
    del model, batch, prediction, loss, graphs, records
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        source = stage_inputs()
        preflight = remote_preflight(source)
        from molgap.qm9_screen import train_encoder

        result = train_encoder(
            candidate="pair_gps_2d_r2",
            geometry="topology",
            train_size=30_000,
            validation_size=3_000,
            test_size=3_000,
            epochs=20,
            seed=42,
            split_seed=42,
            learning_rate=4e-4,
            weight_decay=1e-5,
            patience=8,
            resume=True,
            cache_dir=CACHE,
            results_dir=OUT / "results",
            models_dir=OUT / "models",
        )
        summary = {
            "experiment": "pair_gps_2d_r2_qm9_seed42",
            "source_commit": preflight["source_commit"],
            "preflight": preflight,
            "metrics_path": str(
                OUT
                / "results"
                / "n30000_3000_3000"
                / "pair_gps_2d_r2_topology"
                / "seed42"
                / "metrics.json"
            ),
            "best_epoch": result["best_epoch"],
            "best_validation_average_mae_eV": result[
                "best_validation_average_mae_eV"
            ],
            "test": result["metrics"]["test"],
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_json(OUT / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
