"""Validate that a SCNet DTK/DCU environment can run MolGap graph workloads."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def run_model_smoke_tests(
    torch, device: object, *, include_schnet: bool = True
) -> dict[str, object]:
    """Exercise the exact PyG operators used by the two Phase 8 encoders."""
    from molgap.gps import GPSWrapper
    from molgap.schnet import SchNetWrapper

    report: dict[str, object] = {}
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)

    try:
        gps = GPSWrapper(hidden_channels=16, num_layers=1, num_heads=4, dropout=0.0).to(device)
        gps.eval()
        with torch.no_grad():
            gps_output = gps(
                torch.randn(6, 9, device=device),
                torch.arange(12, dtype=torch.long, device=device).reshape(2, 6) % 6,
                torch.randn(6, 4, device=device),
                batch,
            )
        report["gps_forward_shape"] = list(gps_output.shape)
        report["gps_forward_finite"] = bool(torch.isfinite(gps_output).all().item())
    except Exception as exc:  # Keep the report actionable when a PyG extension is absent.
        report["gps_forward_error"] = f"{type(exc).__name__}: {exc}"

    if not include_schnet:
        report["schnet_forward_skipped"] = True
        return report

    try:
        schnet = SchNetWrapper(
            hidden_channels=16,
            num_filters=16,
            num_interactions=1,
            num_gaussians=16,
            cutoff=5.0,
            dropout=0.0,
        ).to(device)
        schnet.eval()
        with torch.no_grad():
            schnet_output = schnet(
                torch.tensor([6, 6, 8, 6, 7, 6], dtype=torch.long, device=device),
                torch.randn(6, 3, device=device),
                batch,
            )
        report["schnet_forward_shape"] = list(schnet_output.shape)
        report["schnet_forward_finite"] = bool(torch.isfinite(schnet_output).all().item())
    except Exception as exc:  # See above; SchNet specifically reveals radius-graph availability.
        report["schnet_forward_error"] = f"{type(exc).__name__}: {exc}"

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true", help="Permit a local CPU-only check.")
    parser.add_argument(
        "--skip-schnet",
        action="store_true",
        help="Validate the 2D GPS path only when torch-cluster is unavailable.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torch_geometric
    from torch_geometric.nn import global_mean_pool

    accelerator_ready = torch.cuda.is_available()
    device = torch.device("cuda" if accelerator_ready else "cpu")
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_hip_build": getattr(torch.version, "hip", None),
        "torch_geometric": torch_geometric.__version__,
        "accelerator_ready": accelerator_ready,
        "device": str(device),
    }
    if accelerator_ready:
        report["accelerator_name"] = torch.cuda.get_device_name(0)
        report["accelerator_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )

    x = torch.randn(12, 32, device=device)
    batch = torch.tensor([0] * 6 + [1] * 6, device=device)
    pooled = global_mean_pool(x, batch)
    report["pyg_pool_shape"] = list(pooled.shape)
    report["pyg_pool_finite"] = bool(torch.isfinite(pooled).all().item())

    model_smoke = run_model_smoke_tests(torch, device, include_schnet=not args.skip_schnet)
    report["model_smoke"] = model_smoke
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if not accelerator_ready and not args.allow_cpu:
        raise SystemExit("No DTK/DCU accelerator exposed through PyTorch.")
    if not report["pyg_pool_finite"]:
        raise SystemExit("PyG pooling produced non-finite values.")
    if not model_smoke.get("gps_forward_finite"):
        raise SystemExit(f"GPS forward check failed: {model_smoke.get('gps_forward_error')}")
    if not args.skip_schnet and not model_smoke.get("schnet_forward_finite"):
        raise SystemExit(f"SchNet forward check failed: {model_smoke.get('schnet_forward_error')}")


if __name__ == "__main__":
    main()
