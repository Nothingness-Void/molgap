"""Kaggle2: validation-only pure-2D R3 architecture tournament."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/pure2d_r3_validation")
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
CANDIDATES = (
    "edge_state_structural_gps",
    "edge_state_structural_orbital",
    "pair_gps_2d_r3_orbital",
    "pair_gps_2d_r3_triplet",
    "pair_gps_2d_r3_combined",
)
REFERENCE = {
    "candidate": "pair_gps_2d",
    "validation_average_mae_eV": 0.11006919294595718,
    "validation_gap_mae_eV": 0.1318935602903366,
}
PARAMETER_BUDGET = 4_800_000
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
MAX_TOURNAMENT_SECONDS = 6 * 60 * 60


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


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/qm9_screen.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archive = find_one("src.zip")
    extracted = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, extracted)
    modules = list(extracted.rglob("molgap/qm9_screen.py"))
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


def remote_preflight(cache: Path, source_commit: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.loader import DataLoader

    from molgap.qm9_screen import (
        _forward,
        _topology_graph,
        attach_accepted_qm9_rwse,
        configure_frontier_head,
        fixed_split,
        load_qm9_records,
        make_encoder,
        set_seed,
        target_stats,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    records = load_qm9_records(cache)
    split = fixed_split(len(records), 30_000, 3_000, 3_000, 42)
    mean, std = target_stats(records, split.train)
    graphs = {
        "train": [
            _topology_graph(records[int(index)], int(index), mean, std)
            for index in split.train[:48]
        ]
    }
    acceptance = attach_accepted_qm9_rwse(
        graphs, cache_dir=cache, split=split, walk_length=16
    )
    if split.fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            f"Unexpected split fingerprint: {split.fingerprint}"
        )
    if acceptance["output_sha256"] != EXPECTED_RWSE_SHA256:
        raise RuntimeError(
            f"Unexpected RWSE cache: {acceptance['output_sha256']}"
        )
    batch_cpu = next(iter(DataLoader(graphs["train"], batch_size=48)))
    rows = []
    for candidate in CANDIDATES:
        set_seed(42)
        model, kind = make_encoder(candidate)
        head_report = configure_frontier_head(
            model, records, split.train, mean, std
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > PARAMETER_BUDGET:
            raise RuntimeError(
                f"{candidate} parameter budget exceeded: {parameter_count}"
            )
        device = torch.device("cuda")
        model = model.to(device)
        batch = batch_cpu.clone().to(device)
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
        values_eV = prediction.detach() * std.to(device) + mean.to(device)
        consistency = (
            values_eV[:, 1] - values_eV[:, 0] - values_eV[:, 2]
        ).abs().max()
        row = {
            "candidate": candidate,
            "parameter_count": parameter_count,
            "finite_prediction": bool(torch.isfinite(prediction).all()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            ),
            "frontier_head": head_report,
            "max_frontier_identity_error_eV": float(consistency.cpu()),
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_s": time.perf_counter() - started,
        }
        if not all(
            row[key]
            for key in ("finite_prediction", "finite_loss", "finite_gradients")
        ):
            raise RuntimeError(f"Non-finite preflight: {row}")
        if head_report and row["max_frontier_identity_error_eV"] > 1e-5:
            raise RuntimeError(f"Frontier identity preflight failed: {row}")
        rows.append(row)
        del model, batch, prediction, loss, values_eV, gradients
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "format": "molgap-pure2d-r3-preflight-v1",
        "complete": True,
        "source_commit": source_commit,
        "gpu": torch.cuda.get_device_name(0),
        "split_fingerprint": split.fingerprint,
        "rwse_output_sha256": acceptance["output_sha256"],
        "parameter_budget": PARAMETER_BUDGET,
        "validation_role_read": False,
        "test_role_read": False,
        "candidates": rows,
    }
    atomic_json(OUT / "preflight.json", result)
    return result


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ensure_pascal_compatible_torch()
        install_dependencies()
        python_root = source_python_root()
        sys.path.insert(0, str(python_root))
        source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
        acceptance_path = find_one("acceptance.json")
        cache = acceptance_path.parents[2]
        preflight = remote_preflight(cache, source_commit)

        from molgap.qm9_screen import train_encoder

        completed = []
        for candidate in CANDIDATES:
            if time.perf_counter() - started >= MAX_TOURNAMENT_SECONDS:
                raise TimeoutError("Six-hour tournament stop bound reached")
            candidate_started = time.perf_counter()
            result = train_encoder(
                candidate=candidate,
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
                cache_dir=cache,
                results_dir=OUT / "results",
                models_dir=OUT / "models",
                embeddings_dir=OUT / "embeddings",
                evaluate_test=False,
            )
            validation = result["metrics"]["validation"]
            row = {
                "candidate": candidate,
                "parameter_count": result["n_params"],
                "best_epoch": result["best_epoch"],
                "validation": validation,
                "eligible": (
                    validation["average"]["mae"]
                    < REFERENCE["validation_average_mae_eV"]
                    and validation["Gap"]["mae"]
                    < REFERENCE["validation_gap_mae_eV"]
                    and result["n_params"] <= PARAMETER_BUDGET
                ),
                "test_role_evaluated": result["test_role_evaluated"],
                "seconds": time.perf_counter() - candidate_started,
                "model": result["artifacts"]["model"],
                "checkpoint": result["artifacts"]["checkpoint"],
            }
            if row["test_role_evaluated"]:
                raise RuntimeError(f"Validation tournament read test: {candidate}")
            completed.append(row)
            atomic_json(
                OUT / "tournament_progress.json",
                {
                    "source_commit": source_commit,
                    "reference": REFERENCE,
                    "preflight": preflight,
                    "completed": completed,
                    "elapsed_s": time.perf_counter() - started,
                },
            )

        eligible = [row for row in completed if row["eligible"]]
        selected = min(
            eligible,
            key=lambda row: row["validation"]["average"]["mae"],
            default=None,
        )
        selection = {
            "experiment": "pure2d_r3_qm9_validation_tournament",
            "source_commit": source_commit,
            "reference": REFERENCE,
            "parameter_budget": PARAMETER_BUDGET,
            "test_role_read": False,
            "completed": completed,
            "selected_candidate": (
                selected["candidate"] if selected is not None else None
            ),
            "selected_model": selected["model"] if selected is not None else None,
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_json(OUT / "selection.json", selection)
        print(json.dumps(selection, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
