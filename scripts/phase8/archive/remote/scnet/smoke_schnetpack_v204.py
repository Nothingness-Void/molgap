"""Run a bounded SchNetPack 2.0.4 forward/backward gate on ETKDG graphs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        type=Path,
        default=Path("results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt"),
    )
    parser.add_argument("--max-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _make_batch(graphs, neighbor_list, properties, device: torch.device):
    z_parts, pos_parts, center_parts, idx_i_parts, idx_j_parts, offset_parts, y_parts = [], [], [], [], [], [], []
    atom_offset = 0
    for mol_idx, graph in enumerate(graphs):
        z = graph.z.to(device=device, dtype=torch.long)
        pos = graph.pos.to(device=device, dtype=torch.float32)
        inputs = {
            properties.Z: z,
            properties.R: pos,
            properties.cell: torch.zeros((3, 3), device=device, dtype=pos.dtype),
            properties.pbc: torch.zeros(3, device=device, dtype=torch.bool),
        }
        pairs = neighbor_list(inputs)
        z_parts.append(z)
        pos_parts.append(pos)
        center_parts.append(torch.full((z.numel(),), mol_idx, device=device, dtype=torch.long))
        idx_i_parts.append(pairs[properties.idx_i] + atom_offset)
        idx_j_parts.append(pairs[properties.idx_j] + atom_offset)
        offset_parts.append(pairs[properties.offsets])
        y_parts.append(graph.y.view(-1).to(device=device, dtype=torch.float32))
        atom_offset += z.numel()

    return {
        properties.Z: torch.cat(z_parts),
        properties.R: torch.cat(pos_parts),
        properties.idx_i: torch.cat(idx_i_parts),
        properties.idx_j: torch.cat(idx_j_parts),
        properties.offsets: torch.cat(offset_parts),
        properties.idx_m: torch.cat(center_parts),
    }, torch.stack(y_parts)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SchNetPack DCU smoke requires a visible accelerator")

    import schnetpack as spk
    from schnetpack import properties
    from schnetpack.representation import SchNet
    from schnetpack.transform import TorchNeighborList

    device = torch.device("cuda")
    graphs = torch.load(args.graphs, weights_only=False)[: args.max_samples]
    if len(graphs) != args.max_samples:
        raise RuntimeError(f"Expected {args.max_samples} graphs, found {len(graphs)}")

    neighbor_list = TorchNeighborList(cutoff=5.0)
    distances = spk.atomistic.PairwiseDistances().to(device)
    model = SchNet(
        n_atom_basis=64,
        n_interactions=3,
        radial_basis=spk.nn.GaussianRBF(n_rbf=32, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(cutoff=5.0),
    ).to(device)
    head = nn.Linear(64, 3).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    criterion = nn.L1Loss()

    losses, pair_counts = [], []
    t0 = time.time()
    for start in range(0, len(graphs), args.batch_size):
        inputs, target = _make_batch(graphs[start : start + args.batch_size], neighbor_list, properties, device)
        inputs = distances(inputs)
        encoded = model(inputs)["scalar_representation"]
        molecule_count = target.shape[0]
        pooled = torch.zeros((molecule_count, encoded.shape[1]), device=device, dtype=encoded.dtype)
        pooled.index_add_(0, inputs[properties.idx_m], encoded)
        counts = torch.bincount(inputs[properties.idx_m], minlength=molecule_count).clamp_min(1).unsqueeze(1)
        prediction = head(pooled / counts)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite SchNetPack smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pair_counts.append(int(inputs[properties.idx_i].numel()))

    torch.cuda.synchronize()
    report = {
        "schnetpack": importlib.metadata.version("schnetpack"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "graphs": len(graphs),
        "batches": len(losses),
        "batch_size": args.batch_size,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "finite_losses": bool(all(torch.isfinite(torch.tensor(losses)).tolist())),
        "mean_neighbor_pairs_per_graph": sum(pair_counts) / len(graphs),
        "elapsed_s": time.time() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
