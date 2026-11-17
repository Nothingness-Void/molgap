"""Cache Local/GPS embeddings for all from-scratch expert seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR
from molgap.archive.phase8_r04_static_dual2d.models import make_expert
from molgap.archive.phase8_r04_static_dual2d.training import encode_expert


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r04-static-dual2d"


def main() -> None:
    table = pd.read_parquet(OUT_DIR / "pilot_30k.parquet")
    graphs = torch.load(OUT_DIR / "pilot_30k_graphs_2d.pt", weights_only=False)
    graph_map = {int(graph.source_idx.item()): graph for graph in graphs}
    table = table[table.source_idx.isin(graph_map)].reset_index(drop=True)
    ordered = [graph_map[int(index)] for index in table.source_idx]
    loader = DataLoader(ordered, batch_size=256, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in (42, 43, 44):
        arrays = {"source_idx": table.source_idx.to_numpy(dtype=np.int64)}
        for kind in ("local", "global"):
            model = make_expert(kind).to(device)
            model.load_state_dict(torch.load(
                OUT_DIR / f"expert_{kind}" / f"seed{seed}.pt",
                map_location=device,
                weights_only=True,
            ))
            embedding, prediction, indices = encode_expert(kind, model, loader, device)
            if not np.array_equal(indices, arrays["source_idx"]):
                raise ValueError(f"seed {seed} {kind} order mismatch")
            arrays[f"{kind}_embedding"] = embedding
            arrays[f"{kind}_prediction"] = prediction
            del model
            torch.cuda.empty_cache()
        np.savez_compressed(OUT_DIR / f"dual2d_features_seed{seed}.npz", **arrays)
        print(f"cached seed {seed}", flush=True)


if __name__ == "__main__":
    main()
