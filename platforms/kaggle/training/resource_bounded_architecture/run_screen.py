"""Run one three-seed architecture-screen variant on Kaggle."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/resource_bounded_architecture")
RUN_CONFIG = None


def find_one(name: str) -> Path:
    matches = list(INPUT_ROOT.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name}, found {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.check_call(command, env=env)


def ensure_accelerator_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable")
    capability = torch.cuda.get_device_capability(0)
    needs_torch = capability == (6, 0) and "sm_60" not in torch.cuda.get_arch_list()
    try:
        import torch_geometric  # noqa: F401

        needs_pyg = False
    except ModuleNotFoundError:
        needs_pyg = True
    if not needs_torch and not needs_pyg:
        return
    if os.environ.get("MOLGAP_KAGGLE_RUNTIME_RESTART") == "1":
        raise RuntimeError("Kaggle compatibility installation did not take effect")
    if needs_torch:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "--force-reinstall",
                "torch==2.7.1",
                "nvidia-cusparselt-cu12==0.6.3",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "torch-geometric==2.7.0",
        ]
    )
    os.environ["MOLGAP_KAGGLE_RUNTIME_RESTART"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def install_runtime(archive_name: str) -> tuple[Path, dict[str, str]]:
    archives = list(INPUT_ROOT.rglob(archive_name))
    if len(archives) == 1:
        runtime = Path("/kaggle/working/_molgap_resource_bounded_runtime")
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir(parents=True)
        with tarfile.open(archives[0], "r:gz") as handle:
            handle.extractall(runtime)
    else:
        candidates = {
            path.parents[2]
            for path in INPUT_ROOT.rglob("preflight.py")
            if path.parent.name == "resource_bounded_architecture"
            and path.parent.parent.name == "experiments"
        }
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one extracted architecture runtime, found {candidates}"
            )
        source_runtime = candidates.pop()
        runtime = Path("/kaggle/working/_molgap_resource_bounded_runtime")
        if runtime.exists():
            shutil.rmtree(runtime)
        shutil.copytree(source_runtime, runtime)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{runtime / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    return runtime, env


def main() -> None:
    ensure_accelerator_runtime()
    import torch

    if RUN_CONFIG is None:
        raise RuntimeError("packaged RUN_CONFIG is missing")
    config = RUN_CONFIG
    kind = str(config["kind"])
    name = str(config["name"])
    target_mode = str(config.get("target_mode", "all"))
    rwse_alpha_init = float(config.get("rwse_alpha_init", 0.25))
    seeds = [int(seed) for seed in config["seeds"]]
    runtime, env = install_runtime(str(config.get(
        "runtime_archive",
        "resource_bounded_architecture_gap_rwse_r1.tar.gz",
    )))
    base_graph = find_one("pyg_2d_graphs_pubchemqc100k_architecture.pt")
    rwse_graph = find_one("pyg_2d_graphs_pubchemqc100k_rwse16.pt")
    split_csv = find_one("split.csv")
    graph = (
        rwse_graph
        if kind in {
            "structural_gps",
            "normalized_structural_gps",
            "gated_structural_gps",
            "edge_state_structural_gps",
        }
        else base_graph
    )
    output_root = WORK_ROOT / name
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"

    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"variant={name} kind={kind} target_mode={target_mode} seeds={seeds}",
        flush=True,
    )
    preflight = output_root / "preflight.json"
    run(
        [
            sys.executable,
            str(runtime / "experiments/resource_bounded_architecture/preflight.py"),
            "--base-graph",
            str(base_graph),
            "--rwse-graph",
            str(rwse_graph),
            "--split-csv",
            str(split_csv),
            "--output",
            str(preflight),
        ],
        env=env,
    )
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_payload.get("status") != "accepted":
        raise RuntimeError("architecture preflight did not pass")

    accepted: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = output_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        completion = run_dir / "completion_manifest.json"
        if not completion.is_file():
            args = [
                sys.executable,
                str(runtime / "production/03_train/scripts/training/train_encoder.py"),
                "--kind",
                kind,
                "--target-mode",
                target_mode,
                "--graphs",
                str(graph),
                "--split-csv",
                str(split_csv),
                "--hidden-channels",
                "192",
                "--num-layers",
                "9",
                "--num-heads",
                "4",
                "--dropout",
                "0.05",
                "--seed",
                str(seed),
                "--split-seed",
                "42",
                "--epochs",
                "40",
                "--patience",
                "10",
                "--lr",
                "0.0004",
                "--weight-decay",
                "0.00001",
                "--batch-size",
                "256",
                "--eval-batch-size",
                "256",
                "--embedding-batch-size",
                "256",
                "--rwse-dim",
                "16",
                "--no-embeddings",
                "--checkpoint-out",
                str(run_dir / "training_state.pt"),
                "--checkpoint-every",
                "1",
                "--model-out",
                str(run_dir / "model.pt"),
                "--metrics-out",
                str(run_dir / "metrics.json"),
                "--predictions-out",
                str(run_dir / "test_predictions.pt"),
            ]
            if kind == "normalized_structural_gps":
                args.extend(["--rwse-alpha-init", str(rwse_alpha_init)])
            if kind == "edge_state_structural_gps":
                args.extend([
                    "--edge-state-channels",
                    str(int(config.get("edge_state_channels", 64))),
                ])
            checkpoint = run_dir / "training_state.pt"
            if checkpoint.is_file():
                args.extend(["--resume-from", str(checkpoint)])
            run(args, env=env)
            run(
                [
                    sys.executable,
                    str(runtime / "experiments/resource_bounded_architecture/accept_screen_run.py"),
                    "--run-dir",
                    str(run_dir),
                    "--kind",
                    kind,
                    "--seed",
                    str(seed),
                    "--graph",
                    str(graph),
                    "--split-csv",
                    str(split_csv),
                ],
                env=env,
            )
        report = json.loads(completion.read_text(encoding="utf-8"))
        if report.get("status") != "accepted":
            raise RuntimeError(f"seed {seed} lacks accepted completion")
        accepted.append(report)
        atomic_json(
            {
                "format": "molgap-architecture-screen-progress-v1",
                "status": "running" if len(accepted) < len(seeds) else "complete",
                "variant": name,
                "kind": kind,
                "target_mode": target_mode,
                "accepted_seeds": [item["seed"] for item in accepted],
                "expected_seeds": seeds,
                "preflight_sha256": sha256(preflight),
            },
            progress_path,
        )
        print(f"accepted seed={seed}", flush=True)

    atomic_json(
        {
            "format": "molgap-architecture-screen-kaggle-completion-v1",
            "status": "accepted",
            "variant": name,
            "kind": kind,
            "target_mode": target_mode,
            "seeds": seeds,
            "input_sha256": {
                "base_graph": sha256(base_graph),
                "rwse_graph": sha256(rwse_graph),
                "split_csv": sha256(split_csv),
            },
            "preflight": preflight_payload,
            "runs": accepted,
        },
        output_root / "kernel_completion_manifest.json",
    )
    print(f"COMPLETE variant={name} accepted_seeds={seeds}", flush=True)


if __name__ == "__main__":
    main()
