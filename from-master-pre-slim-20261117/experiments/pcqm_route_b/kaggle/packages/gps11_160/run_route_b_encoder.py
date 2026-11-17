"""Kaggle adapter for one PCQM Route B streaming encoder continuation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENCODER_NAME = "gps11_160"
WARM_START_NAME = "gps11_160_repaired_2m_seed42.pt"


def ensure_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
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
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def find_view(modality: str) -> Path:
    matches = {
        path.parent.parent
        for path in Path("/kaggle/input").rglob(
            f"{modality}/train_shard_*.pt"
        )
    }
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one mounted {modality} graph root, found {sorted(matches)}"
        )
    return matches.pop()


def install_local_package() -> None:
    root = Path("/kaggle/working/_molgap_runtime")
    package = root / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("gps.py", "schnet.py", "pcqm_route_b_training.py"):
        shutil.copy2(find_one(name), package / name)
    sys.path.insert(0, str(root))


def main() -> None:
    if ENCODER_NAME is None or WARM_START_NAME is None:
        raise RuntimeError("package_variants.py must embed the encoder contract")
    ensure_runtime()
    install_local_package()
    import torch
    from molgap.pcqm_route_b_training import CONFIGS, train_encoder

    config = CONFIGS[ENCODER_NAME]
    roots = {config.modality: find_view(config.modality)}
    if config.augmented:
        roots["secondary"] = find_view("secondary")
    print(
        f"encoder={config.name} gpu={torch.cuda.get_device_name(0)} "
        f"roots={roots}",
        flush=True,
    )
    result = train_encoder(
        config=config,
        roots=roots,
        warm_start=find_one(WARM_START_NAME),
        output_dir=Path("/kaggle/working") / config.name,
    )
    print(
        f"complete encoder={config.name} best_epoch={result['best_epoch']} "
        f"dev_gap_mae={result['best_dev_gap_mae_eV']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
