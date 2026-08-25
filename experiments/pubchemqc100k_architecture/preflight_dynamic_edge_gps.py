"""Compute-node forward/backward smoke test for DynamicEdgeGPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from molgap.edge_state_gps import DynamicEdgeGPSWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SCNet preflight did not expose a DCU accelerator")
    device = torch.device("cuda")
    graphs = torch.load(args.graphs, weights_only=False)[:4]
    model = DynamicEdgeGPSWrapper(
        in_channels=9,
        edge_dim=4,
        hidden_channels=160,
        num_layers=11,
        num_heads=4,
        dropout=0.05,
        n_targets=3,
        max_degree=8,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
    batch = next(iter(DataLoader(graphs, batch_size=4, shuffle=False, num_workers=0))).to(device)
    with torch.amp.autocast("cuda"):
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = torch.nn.functional.l1_loss(output, batch.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    report = {
        "device": torch.cuda.get_device_name(0),
        "graphs": len(graphs),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(json.dumps(report, indent=2), flush=True)
    if not report["output_finite"] or not report["loss_finite"]:
        raise RuntimeError("DynamicEdgeGPS preflight produced non-finite values")


if __name__ == "__main__":
    main()

