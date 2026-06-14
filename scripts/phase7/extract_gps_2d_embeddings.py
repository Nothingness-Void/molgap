"""
Phase 7: Fast standalone 2D embedding extraction from trained GPS 2D model.

Loads models/gps_2d_300k.pt, runs every 2D graph through encode() (pooled 192-d
vector), saves results/phase7/gps_2d_embeddings.pt.

num_workers=0 + large batch + progress print — avoids the Windows DataLoader
worker-spawn stall that made the in-script extraction hang.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/extract_gps_2d_embeddings.py
"""
from __future__ import annotations

import time

import torch
from torch_geometric.loader import DataLoader

from molgap.constants import MODELS_DIR, RESULTS_DIR
from molgap.gps import GPSWrapper

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_2d_graphs_bond_300k.pt"
MODEL_PATH = MODELS_DIR / "gps_2d_300k.pt"
OUT_PATH = PHASE7_DIR / "gps_2d_embeddings.pt"

BATCH_SIZE = 1024

# Must match the trained model (Kaggle Optuna best params)
BP = {
    "hidden_channels": 192,
    "num_layers": 7,
    "num_heads": 4,
    "dropout": 0.05,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graphs from {GRAPH_PATH} ...")
    graphs = torch.load(str(GRAPH_PATH), weights_only=False)
    print(f"Loaded {len(graphs)} 2D graphs")

    model = GPSWrapper(
        hidden_channels=BP["hidden_channels"],
        num_layers=BP["num_layers"],
        num_heads=BP["num_heads"],
        dropout=BP["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load(str(MODEL_PATH), weights_only=False, map_location=device)
    )
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_batches = (len(graphs) + BATCH_SIZE - 1) // BATCH_SIZE

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch_data in enumerate(loader):
            batch_data = batch_data.to(device)
            with torch.amp.autocast("cuda"):
                emb = model.encode(batch_data.x, batch_data.edge_index,
                                   batch_data.edge_attr, batch_data.batch)
            embeddings.append(emb.float().cpu())
            if bi % 10 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                print(f"  batch {bi+1}/{n_batches}  ({elapsed:.0f}s)", flush=True)

    embeddings = torch.cat(embeddings, dim=0)
    print(f"\n2D embeddings: {embeddings.shape}")
    torch.save(embeddings, str(OUT_PATH))
    print(f"Saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
