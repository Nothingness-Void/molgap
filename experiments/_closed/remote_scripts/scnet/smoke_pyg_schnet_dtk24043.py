"""Bounded full MolGap SchNet smoke for the DTK 24.04.3 PyG stack."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from molgap.schnet import SchNetWrapper


def main() -> None:
    root = Path.home() / "molgap"
    graphs_path = root / "results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt"
    output_dir = Path.home() / "scnet-molgap-experiments/20260717_phase9_2d/schnet3d_dtk24043_smoke"

    if not torch.cuda.is_available():
        raise RuntimeError("No DCU visible")

    graphs = torch.load(graphs_path, weights_only=False)[:64]
    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    device = torch.device("cuda")
    model = SchNetWrapper(
        hidden_channels=64,
        num_filters=64,
        num_interactions=3,
        num_gaussians=32,
        cutoff=5.0,
        n_targets=3,
        use_charges=hasattr(graphs[0], "charges"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    started = time.time()
    for batch in loader:
        batch = batch.to(device)
        charges = getattr(batch, "charges", None)
        prediction = model(batch.z, batch.pos, batch.batch, charges=charges)
        target = batch.y.view(prediction.shape[0], -1).to(prediction.dtype)
        loss = torch.nn.functional.l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    torch.cuda.synchronize()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "graphs": len(graphs),
        "batches": len(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "elapsed_s": time.time() - started,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
