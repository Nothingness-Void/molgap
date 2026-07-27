"""Kaggle entrypoint for one PubChemQC 100K lightweight SchNet variant."""

from __future__ import annotations

import os
import subprocess
import sys
import shutil
from pathlib import Path

VARIANT = "augmented"
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def ensure_pascal_compatible_torch() -> None:
    """Replace Kaggle's stock torch when it cannot execute on a P100."""
    import torch as probe_torch

    if not probe_torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
    capability = probe_torch.cuda.get_device_capability(0)
    supported_arches = set(probe_torch.cuda.get_arch_list())
    if capability != (6, 0) or "sm_60" in supported_arches:
        return
    if os.environ.get(PASCAL_COMPAT_RESTART) == "1":
        raise RuntimeError(
            "The cu126 compatibility install still lacks sm_60 support; "
            "refusing a restart loop"
        )
    print(
        "Kaggle assigned a P100 but stock torch lacks sm_60; "
        "installing torch 2.7.1+cu126 once.",
        flush=True,
    )
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


def install_pyg_extensions() -> None:
    import torch

    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    # PyG publishes one extension index for the torch 2.7 patch line.
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    wheel_index = (
        f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "-f",
            wheel_index,
        ]
    )
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"wheel_index={wheel_index}",
        flush=True,
    )


def validate_gpu_runtime() -> None:
    import torch
    from torch_cluster import radius_graph

    device = torch.device("cuda")
    probe = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    edges = radius_graph(
        probe,
        r=2.0,
        batch=torch.zeros(2, dtype=torch.long, device=device),
    )
    if edges.shape != (2, 2):
        raise RuntimeError(f"radius_graph GPU preflight failed: {edges}")
    print(
        f"gpu={torch.cuda.get_device_name(0)} capability="
        f"{torch.cuda.get_device_capability(0)} radius_graph=OK",
        flush=True,
    )


def main() -> None:
    if VARIANT not in {"primary", "augmented"}:
        raise RuntimeError("VARIANT must be embedded by package_variants.py")
    ensure_pascal_compatible_torch()
    install_pyg_extensions()
    validate_gpu_runtime()
    import torch

    runtime_root = Path("/kaggle/working/_molgap_runtime")
    package = runtime_root / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(find_one("pubchemqc_architecture.py"), package)
    shutil.copy2(find_one("schnet.py"), package)
    sys.path.insert(0, str(runtime_root))
    from molgap.pubchemqc_architecture import train

    primary_path = find_one("pyg_3d_graphs_etkdg_expansion_1m.pt")
    split_path = find_one("split.csv")
    secondary_paths = sorted(Path("/kaggle/input").rglob("graphs_*.pt"))
    if len(secondary_paths) != 24:
        raise RuntimeError(
            f"Expected 24 accepted secondary graph parts, found {len(secondary_paths)}"
        )
    print(f"loading primary graph cache: {primary_path}", flush=True)
    primary = torch.load(primary_path, map_location="cpu", weights_only=False)
    secondary = []
    for path in secondary_paths:
        part = torch.load(path, map_location="cpu", weights_only=False)
        secondary.extend(part)
        print(
            f"loaded secondary {path.name}: part={len(part)} total={len(secondary)}",
            flush=True,
        )
    print(
        f"inputs primary={len(primary)} secondary={len(secondary)} "
        f"split={split_path}",
        flush=True,
    )
    result = train(
        variant=VARIANT,
        primary_graphs=primary,
        secondary_graphs=secondary,
        split_csv=split_path,
        output_dir=Path("/kaggle/working") / f"light_schnet_{VARIANT}",
        epochs=30,
        patience=8,
        batch_size=128,
        learning_rate=4e-4,
        weight_decay=1e-5,
        seed=42,
    )
    print(
        f"complete variant={VARIANT} best_epoch={result['best_epoch']} "
        f"validation={result['best_validation_average_mae_eV']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
