"""Kaggle tuned-1M primary SchNet fallback for the freeze sprint."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


ENCODER = "primary_schnet"
WINNER = {
    "trial_id": "trial_11",
    "learning_rate": 0.0006,
    "weight_decay": 3e-6,
    "dropout": 0.05,
    "batch_size": 64,
    "warmup_ratio": 0.1,
    "grad_clip": 2.0,
}


def ensure_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability == (6, 0) and "sm_60" not in torch.cuda.get_arch_list():
        if os.environ.get("MOLGAP_TORCH_COMPAT_RESTART") == "1":
            raise RuntimeError("P100 compatibility install did not provide sm_60")
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
        os.environ["MOLGAP_TORCH_COMPAT_RESTART"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "-f",
            f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html",
        ]
    )


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name}, found {matches}")
    return matches[0]


def find_view(modality: str) -> Path:
    roots = {
        path.parent.parent
        for path in Path("/kaggle/input").rglob(f"{modality}/train_shard_*.pt")
    }
    if len(roots) != 1:
        raise FileNotFoundError(f"expected one {modality} graph root: {roots}")
    return roots.pop()


def install_local_package() -> None:
    root = Path("/kaggle/working/_molgap_runtime")
    package = root / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    sources = {
        "tuned_gps.py": "gps.py",
        "tuned_schnet.py": "schnet.py",
        "tuned_pcqm_route_b_training.py": "pcqm_route_b_training.py",
    }
    for source_name, target_name in sources.items():
        shutil.copy2(find_one(source_name), package / target_name)
    sys.path.insert(0, str(root))


def main() -> None:
    ensure_runtime()
    install_local_package()
    import torch
    from molgap.pcqm_route_b_training import (
        CONFIGS,
        TrainingOverrides,
        preflight_encoder,
        train_encoder,
    )

    graph_root = find_view("primary")
    warm_start = find_one("primary_schnet_100k.pt")
    config = replace(
        CONFIGS[ENCODER],
        batch_size=WINNER["batch_size"],
        learning_rate=WINNER["learning_rate"],
        weight_decay=WINNER["weight_decay"],
    )
    overrides = TrainingOverrides(
        dropout=WINNER["dropout"],
        warmup_ratio=WINNER["warmup_ratio"],
        grad_clip=WINNER["grad_clip"],
        schnet_cutoff=6.0,
    )
    print(
        f"gpu={torch.cuda.get_device_name(0)} encoder={ENCODER} "
        f"winner={WINNER['trial_id']}",
        flush=True,
    )
    output_dir = Path("/kaggle/working/tuned_primary_schnet")
    preflight_encoder(
        config=config,
        root=graph_root,
        warm_start=warm_start,
        output_path=output_dir / "preflight",
        overrides=overrides,
    )
    result = train_encoder(
        config=config,
        roots={"primary": graph_root},
        warm_start=warm_start,
        output_dir=output_dir,
        overrides=overrides,
    )
    print(
        f"complete best_epoch={result['best_epoch']} "
        f"dev_gap_mae={result['best_dev_gap_mae_eV']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
