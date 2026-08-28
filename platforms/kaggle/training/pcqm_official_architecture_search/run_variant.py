"""Run one bounded official-train-only PCQM architecture candidate."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


RUN_CONFIG = None
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/pcqm_official_architecture_search")
EXPECTED_ACCEPTANCE_SHA256 = (
    "faecf13321e373e76216e7cd6a6ab64e826d6983d2e595f419874e410d0bb3a4"
)
CONTROL = {
    "mae_eV": 0.15006214380264282,
    "radical_mae_eV": 0.25360438227653503,
    "nonradical_mae_eV": 0.1414971500635147,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_runtime() -> Path:
    if not importlib.metadata.version("torch").startswith("2.5.1"):
        print("Installing P100-compatible torch 2.5.1+cu121", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "--force-reinstall",
                "torch==2.5.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "torch-geometric==2.6.1",
            ]
        )
    try:
        import rdkit  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "rdkit==2025.3.6",
            ]
        )

    matches = sorted(INPUT_ROOT.rglob("pcqm_feature_screen.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one feature runtime, found {matches}")
    runtime = Path("/kaggle/working/pcqm_architecture_runtime")
    package = runtime / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    for source in matches[0].parent.glob("*.py"):
        shutil.copy2(source, package / source.name)
    sys.path.insert(0, str(runtime))
    return runtime


def main() -> None:
    if RUN_CONFIG is None:
        raise RuntimeError("packaged RUN_CONFIG is missing")
    prepare_runtime()

    import torch
    from molgap.pcqm_feature_screen import (
        FeatureScreenConfig,
        preflight_feature_model,
        train_feature_screen,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable")
    acceptance_matches = [
        path
        for path in INPUT_ROOT.rglob("acceptance.json")
        if sha256(path) == EXPECTED_ACCEPTANCE_SHA256
    ]
    if len(acceptance_matches) != 1:
        raise RuntimeError(
            f"expected one accepted graph mount, found {acceptance_matches}"
        )
    acceptance = acceptance_matches[0]
    graph_dir = acceptance.parent
    variant = str(RUN_CONFIG["variant"])
    seed = int(RUN_CONFIG["seed"])
    output = OUTPUT_ROOT / f"{variant}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = FeatureScreenConfig(
        model_family=str(RUN_CONFIG.get("model_family", "edge_state")),
        precision=str(RUN_CONFIG.get("precision", "amp")),
        hidden_channels=int(RUN_CONFIG["hidden_channels"]),
        num_layers=int(RUN_CONFIG["num_layers"]),
        num_heads=int(RUN_CONFIG["num_heads"]),
        edge_state_channels=int(RUN_CONFIG["edge_state_channels"]),
        categorical_encoder=str(RUN_CONFIG.get("categorical_encoder", "sum")),
        categorical_field_channels=int(
            RUN_CONFIG.get("categorical_field_channels", 16)
        ),
        graph_context=str(RUN_CONFIG.get("graph_context", "none")),
        radical_context_channels=int(
            RUN_CONFIG.get("radical_context_channels", 16)
        ),
        pair_channels=int(RUN_CONFIG.get("pair_channels", 64)),
        path_steps=int(RUN_CONFIG.get("path_steps", 5)),
        triplet_rank=int(RUN_CONFIG.get("triplet_rank", 16)),
        atom_input_channels=int(RUN_CONFIG.get("atom_input_channels", 64)),
        bond_input_channels=int(RUN_CONFIG.get("bond_input_channels", 32)),
        batch_size=int(RUN_CONFIG.get("batch_size", 256)),
        eval_batch_size=int(RUN_CONFIG.get("eval_batch_size", 512)),
        max_epochs=20,
        scheduler="warmup_cosine",
        warmup_epochs=2,
        patience=7,
        hard_job_budget_s=10.0 * 3600.0,
    )
    print(
        f"device={torch.cuda.get_device_name(0)} variant={variant} seed={seed} "
        f"config={config}",
        flush=True,
    )
    preflight = preflight_feature_model(
        graph_dir,
        acceptance,
        output / "preflight.json",
        schema="ogb",
        config=config,
        batch_size=config.batch_size,
    )
    if preflight["projected_training_hours"] > 10.0:
        raise RuntimeError(
            "candidate exceeds the ten-hour preflight gate: "
            f"{preflight['projected_training_hours']:.2f} h"
        )
    print(f"preflight={preflight}", flush=True)
    if RUN_CONFIG.get("mode", "train") == "preflight":
        completion = {
            "format": "molgap-pcqm-official-architecture-preflight-kaggle-v1",
            "status": "complete",
            "variant": variant,
            "seed": seed,
            "official_train_only": True,
            "official_valid_used": False,
            "official_test_used": False,
            "external_data_used": False,
            "dataset": "nothingnessvoid/molgap-pcqm-feature-screen-20260826",
            "acceptance_sha256": sha256(acceptance),
            "config": RUN_CONFIG,
            "preflight": preflight,
            "artifacts": {
                "preflight.json": {
                    "bytes": (output / "preflight.json").stat().st_size,
                    "sha256": sha256(output / "preflight.json"),
                }
            },
        }
        atomic_json(output / "kernel_completion_manifest.json", completion)
        print(json.dumps(completion, indent=2), flush=True)
        return
    metrics = train_feature_screen(
        graph_dir,
        acceptance,
        output,
        schema="ogb",
        seed=seed,
        config=config,
    )
    development = metrics["development"]
    deltas = {
        key: float(development[key]) - value for key, value in CONTROL.items()
    }
    gate = {
        "overall_improvement_at_least_0p002_eV": deltas["mae_eV"] <= -0.002,
        "radical_regression_at_most_0p002_eV": deltas["radical_mae_eV"] <= 0.002,
        "nonradical_regression_at_most_0p002_eV": deltas[
            "nonradical_mae_eV"
        ]
        <= 0.002,
    }
    artifacts = {}
    for name in (
        "best.pt",
        "last.pt",
        "metrics.json",
        "development_predictions.pt",
        "completion_manifest.json",
        "progress.json",
        "preflight.json",
    ):
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    completion = {
        "format": "molgap-pcqm-official-architecture-kaggle-v1",
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "official_train_only": True,
        "official_valid_used": False,
        "official_test_used": False,
        "external_data_used": False,
        "dataset": "nothingnessvoid/molgap-pcqm-feature-screen-20260826",
        "acceptance_sha256": sha256(acceptance),
        "config": RUN_CONFIG,
        "development": development,
        "control": CONTROL,
        "delta_eV": deltas,
        "provisional_gate": gate,
        "provisional_gate_pass": all(gate.values()),
        "artifacts": artifacts,
    }
    atomic_json(output / "kernel_completion_manifest.json", completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
