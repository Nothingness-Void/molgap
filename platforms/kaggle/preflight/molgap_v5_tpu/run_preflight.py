from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "9c0ed4ee5389e2c6b79e17e68beba34df31986d422a112f1c661e87e5adb1913"
)
EXPECTED_RUNTIME_COMMIT = "24d9b13fad48bab1d0a8a92fec2b0b88b57224c9"
EXPECTED_DATA_MANIFEST_SHA256 = (
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
)
OUTPUT = Path("/kaggle/working/tpu_preflight.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def locate_manifest(expected_sha256: str, name: str) -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob(name):
        if path.is_file() and sha256_file(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} with SHA256 {expected_sha256}; "
            f"found {matches}"
        )
    return matches[0]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def main() -> None:
    report = {
        "format": "molgap-v5-kaggle-tpu-preflight-v1",
        "status": "running",
        "started_at_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "phases": {},
    }
    atomic_json(OUTPUT, report)
    try:
        runtime_manifest = locate_manifest(
            EXPECTED_RUNTIME_MANIFEST_SHA256, "runtime_manifest.json"
        )
        runtime_root = runtime_manifest.parent / "molgap_runtime"
        runtime = json.loads(runtime_manifest.read_text())
        if runtime["source_commit"] != EXPECTED_RUNTIME_COMMIT:
            raise RuntimeError("Mounted runtime commit mismatch")
        sys.path.insert(0, str(runtime_root / "src"))

        data_manifest = locate_manifest(
            EXPECTED_DATA_MANIFEST_SHA256, "manifest.json"
        )
        data = json.loads(data_manifest.read_text())
        if data["identity"]["name"] != "ogb-train-100k":
            raise RuntimeError("Mounted fixed-data identity mismatch")
        first_shard = data_manifest.parent / data["geometry_shards"][0]["file"]
        if first_shard.stat().st_size != data["geometry_shards"][0]["bytes"]:
            raise RuntimeError("Mounted first-shard size mismatch")
        report["phases"]["mounts"] = {
            "status": "passed",
            "runtime_manifest": str(runtime_manifest),
            "runtime_source_commit": runtime["source_commit"],
            "data_manifest": str(data_manifest),
            "data_identity": data["identity"],
            "first_shard": str(first_shard),
        }
        atomic_json(OUTPUT, report)

        jax_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import jax,json; print(json.dumps([str(x) for x in jax.devices()]))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        jax_devices = json.loads(jax_probe.stdout.strip().splitlines()[-1])
        report["phases"]["jax_devices"] = {
            "status": "passed" if len(jax_devices) == 8 else "failed",
            "count": len(jax_devices),
            "devices": jax_devices,
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
        }
        atomic_json(OUTPUT, report)

        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        x = torch.randn(128, 64, device=device)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        ).to(device)
        started = time.perf_counter()
        loss = model(x).square().mean()
        loss.backward()
        xm.mark_step()
        dense_loss = float(loss.detach().cpu())
        report["phases"]["torch_xla_dense"] = {
            "status": "passed",
            "device": str(device),
            "torch": torch.__version__,
            "torch_xla": getattr(torch_xla, "__version__", version("torch-xla")),
            "loss": dense_loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(OUTPUT, report)

        import torch_geometric
        from torch.utils.data import Subset
        from torch_geometric.data import InMemoryDataset
        from torch_geometric.loader import DataLoader

        from molgap.gptrans import OGBGPTransTiny

        class PackedGraphs(InMemoryDataset):
            def __init__(self, path: Path):
                super().__init__(root=None)
                self.data, self.slices = torch.load(
                    path, map_location="cpu", weights_only=False
                )

        graphs = PackedGraphs(first_shard)
        batch = next(iter(DataLoader(Subset(graphs, range(8)), batch_size=8)))
        graph_model = OGBGPTransTiny(
            node_channels=64,
            pair_channels=16,
            num_layers=2,
            num_heads=4,
            shortest_path_cap=20,
            dropout=0.0,
            drop_path=0.0,
            layer_scale=1.0,
            n_targets=1,
        ).to(device)
        batch = batch.to(device)
        started = time.perf_counter()
        prediction = graph_model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            getattr(batch, "random_walk_pe", None),
        ).view(-1)
        target = batch.y.view(-1).to(prediction.dtype)
        graph_loss = torch.nn.functional.l1_loss(prediction, target)
        graph_loss.backward()
        xm.mark_step()
        graph_loss_value = float(graph_loss.detach().cpu())
        report["phases"]["pyg_gptrans_backward"] = {
            "status": "passed",
            "torch_geometric": torch_geometric.__version__,
            "graphs": 8,
            "parameters": sum(p.numel() for p in graph_model.parameters()),
            "loss": graph_loss_value,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report["status"] = (
            "passed"
            if report["phases"]["jax_devices"]["status"] == "passed"
            else "failed"
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    report["finished_at_unix"] = time.time()
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
