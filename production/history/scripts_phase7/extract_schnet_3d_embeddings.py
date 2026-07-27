"""
Phase 7: extract 3D molecular embeddings from the trained SchNet 300k model.

Runs every 3D ETKDG graph through SchNetWrapper.encode() (pooled 192-d vector,
pre-head), saves results/phase7/schnet_3d_embeddings.pt for hybrid fusion.

num_workers=0 + large batch + progress print (Windows-friendly).

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/extract_schnet_3d_embeddings.py
"""
from __future__ import annotations

import time

import torch
from torch_geometric.loader import DataLoader

from molgap.constants import MODELS_DIR, RESULTS_DIR
from molgap.schnet import SchNetWrapper

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"
MODEL_PATH = MODELS_DIR / "gnn_schnet_3d_300k.pt"
OUT_PATH = PHASE7_DIR / "schnet_3d_embeddings.pt"

BATCH_SIZE = 256

PARAMS_300K = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graphs from {GRAPH_PATH} ...")
    graphs = torch.load(str(GRAPH_PATH), weights_only=False)
    print(f"Loaded {len(graphs)} 3D graphs")

    model = SchNetWrapper(**PARAMS_300K, use_charges=True).to(device)
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
        for bi, batch in enumerate(loader):
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            with torch.amp.autocast("cuda"):
                emb = model.encode(batch.z, batch.pos, batch.batch, charges=charges)
            embeddings.append(emb.float().cpu())
            if bi % 50 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                print(f"  batch {bi+1}/{n_batches}  ({elapsed:.0f}s)", flush=True)

    embeddings = torch.cat(embeddings, dim=0)
    print(f"\n3D embeddings: {embeddings.shape}")
    torch.save(embeddings, str(OUT_PATH))
    print(f"Saved to {OUT_PATH}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
