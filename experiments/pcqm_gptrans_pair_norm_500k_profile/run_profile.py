"""Bounded Kaggle3 T4 profile for the GPTrans pair-update-normalization path."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


EXPECTED_MANIFEST_SHA256 = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
EXPECTED_SHARD_SHA256 = "8cc4c6373c22c8f72ac302f514bb657c5b40a9a1c620c6228b760380ebec705e"
EXPECTED_SOURCE_SHA256 = "8b034350a7188cccc3c9b112625d874af910bc8034ea6591f1607f8e85d47ccf"
EXPECTED_PARAMETERS = 5_246_817
BATCH_SIZE = 128
WARMUP_STEPS = 10
MEASURE_STEPS = 80
TRAIN_ROWS = 500_000
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE
EPOCHS = 60
CASES = (
    {"name": "workers0", "workers": 0, "prefetch_factor": None},
    {"name": "workers2_pf2", "workers": 2, "prefetch_factor": 2},
    {"name": "workers4_pf2", "workers": 4, "prefetch_factor": 2},
    {"name": "workers4_pf4", "workers": 4, "prefetch_factor": 4},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_unique(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


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


def configure_torch(torch) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_shard(torch, shard_path: Path):
    from torch_geometric.data import InMemoryDataset

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(
                path, map_location="cpu", weights_only=False
            )

    dataset = PackedGraphDataset(shard_path)
    if len(dataset) != 50_000:
        raise RuntimeError(f"Representative shard has {len(dataset)} rows")
    return dataset


def make_loader(dataset, case):
    import torch
    from torch_geometric.loader import DataLoader

    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "drop_last": True,
        "num_workers": case["workers"],
        "pin_memory": True,
        "persistent_workers": case["workers"] > 0,
        "generator": torch.Generator().manual_seed(42),
    }
    if case["workers"] > 0:
        kwargs["prefetch_factor"] = case["prefetch_factor"]
    return DataLoader(**kwargs)


def percentile(values, q):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def run_case(torch, dataset, case, initial_state_path: Path) -> dict:
    from molgap.pcqm_gptrans_v4 import _forward, _make_model

    configure_torch(torch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = _make_model(initial_state_path=initial_state_path, variant="pair_update_norm")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Model parameter identity changed")
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, weight_decay=1e-5, foreach=False, fused=False
    )
    loader = make_loader(dataset, case)
    iterator = iter(loader)

    def one_step(measure: bool):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        batch = next(iterator)
        loaded = time.perf_counter()
        if batch.num_graphs != BATCH_SIZE:
            raise RuntimeError(f"Physical batch changed: {batch.num_graphs}")
        batch = batch.cuda(non_blocking=True)
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, batch)
        target = batch.y.view(-1)
        loss = torch.nn.functional.l1_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite profile loss")
        if not measure:
            return None
        return {
            "loader_seconds": loaded - wall_start,
            "h2d_seconds": transferred - loaded,
            "compute_seconds": finished - transferred,
            "step_seconds": finished - wall_start,
            "loss": float(loss.detach().cpu()),
        }

    for _ in range(WARMUP_STEPS):
        one_step(False)
    measurements = [one_step(True) for _ in range(MEASURE_STEPS)]
    step_seconds = [item["step_seconds"] for item in measurements]
    median_step = statistics.median(step_seconds)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    result = {
        **case,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURE_STEPS,
        "batch_size": BATCH_SIZE,
        "median_step_seconds": median_step,
        "p90_step_seconds": percentile(step_seconds, 0.90),
        "mean_loader_seconds": statistics.mean(item["loader_seconds"] for item in measurements),
        "mean_h2d_seconds": statistics.mean(item["h2d_seconds"] for item in measurements),
        "mean_compute_seconds": statistics.mean(item["compute_seconds"] for item in measurements),
        "graphs_per_second": BATCH_SIZE / median_step,
        "projected_epoch_seconds": STEPS_PER_EPOCH * median_step,
        "projected_60_epoch_hours": STEPS_PER_EPOCH * median_step * EPOCHS / 3600,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "total_device_bytes": total_memory,
        "reserved_fraction": torch.cuda.max_memory_reserved() / total_memory,
        "last_loss": measurements[-1]["loss"],
    }
    del iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    output = Path("/kaggle/working/profile")
    output.mkdir(parents=True, exist_ok=True)
    install_dependencies()
    import torch

    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"Expected Kaggle T4x2, found {gpu_names}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.device_count() != 2:
        raise RuntimeError("CUDA allocation changed before profiling")
    torch.cuda.set_device(0)
    configure_torch(torch)

    manifest_path = find_unique("manifest.json")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Fixed 500K manifest identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity"]["name"] != "ogb-train-500k-scnet-v1":
        raise RuntimeError("Wrong fixed-data role")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Sealed role flag changed: {role}")
    shard_path = manifest_path.parent / "train/train_shard_0000.pt"
    if sha256_file(shard_path) != EXPECTED_SHARD_SHA256:
        raise RuntimeError("Representative shard identity changed")

    source_root = find_unique("src/molgap/gptrans_variants.py").parents[2]
    sys.path.insert(0, str(source_root / "src"))
    source_sha_path = find_unique("SOURCE_ARCHIVE_SHA256.txt")
    if source_sha_path.read_text(encoding="utf-8").strip() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Accepted pair-normalization source identity changed")
    initial_state_path = find_unique("initial_state.pt")
    dataset = load_shard(torch, shard_path)

    report = {
        "format": "molgap-gptrans-pair-update-500k-profile-v1",
        "status": "running",
        "model": "gptrans_t_core_12x256_pair32/pair_update_norm",
        "parameters": EXPECTED_PARAMETERS,
        "hardware": {"allocated": gpu_names, "profile_device": gpu_names[0]},
        "software": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "shard_sha256": EXPECTED_SHARD_SHA256,
            "rows_read": len(dataset),
            "development_role_read": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
        "execution": {
            "precision": "fp32",
            "tf32_enabled": False,
            "deterministic_algorithms": True,
            "physical_batch_per_device": BATCH_SIZE,
            "active_device_count": 1,
            "optimizer": "adamw-unfused-foreachFalse",
        },
        "cases": [],
    }
    atomic_json(output / "profile.json", report)
    for case in CASES:
        result = run_case(torch, dataset, case, initial_state_path)
        report["cases"].append(result)
        atomic_json(output / "profile.json", report)
        print(
            f"PROFILE {case['name']} graphs_per_second={result['graphs_per_second']:.2f} "
            f"projected_60ep_hours={result['projected_60_epoch_hours']:.2f} "
            f"reserved={result['reserved_fraction']:.3f}",
            flush=True,
        )

    eligible = [case for case in report["cases"] if case["reserved_fraction"] <= 0.85]
    if len(eligible) != len(CASES):
        raise RuntimeError("At least one profile case exceeded the memory gate")
    winner = max(eligible, key=lambda item: item["graphs_per_second"])
    baseline = next(item for item in eligible if item["name"] == "workers0")
    report["status"] = "complete"
    report["recommendation"] = {
        "case": winner["name"],
        "speedup_vs_workers0": winner["graphs_per_second"] / baseline["graphs_per_second"],
        "projected_epoch_seconds": winner["projected_epoch_seconds"],
        "projected_60_epoch_hours": winner["projected_60_epoch_hours"],
        "planning_estimate_only": True,
        "authorizes_training": False,
    }
    atomic_json(output / "profile.json", report)
    print("PROFILE COMPLETE " + json.dumps(report["recommendation"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
