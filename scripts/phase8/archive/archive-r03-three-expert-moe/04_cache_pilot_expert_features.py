"""Cache frozen expert embeddings, outputs, and Router descriptors."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR
from molgap.gps import GPSWrapper
from molgap.pubchemqc import sha256_file
from molgap.router import router_descriptor_row
from molgap.schnet import SchNetWrapper
from molgap.dual2d_static_candidate.local_gine import LocalGINEExpert
from molgap.dual2d_static_candidate.training import encode_expert


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r03-three-expert-moe"
DUAL2D_DIR = RESULTS_DIR / "phase8" / "dual2d_static_candidate"
EXPERTS = ("local", "global", "geometry")


def model_for(kind):
    if kind == "local":
        return LocalGINEExpert()
    if kind == "global":
        return GPSWrapper(
            hidden_channels=192, num_layers=9, num_heads=4,
            dropout=0.05, pooling="mean_max",
        )
    return SchNetWrapper(
        hidden_channels=192, num_filters=192, num_interactions=6,
        num_gaussians=50, cutoff=6.0, dropout=0.05,
    )


def main() -> None:
    table = pd.read_parquet(DUAL2D_DIR / "pilot_30k.parquet")
    graph_maps = {}
    for space, name in (("2d", "pilot_30k_graphs_2d.pt"), ("3d", "pilot_30k_graphs_3d.pt")):
        source_dir = DUAL2D_DIR if space == "2d" else OUT_DIR
        graphs = torch.load(source_dir / name, weights_only=False)
        graph_maps[space] = {int(graph.source_idx.item()): graph for graph in graphs}
    common = set(graph_maps["2d"]) & set(graph_maps["3d"])
    table = table[table.source_idx.isin(common)].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = {"source_idx": table.source_idx.to_numpy(dtype=np.int64)}
    for kind in EXPERTS:
        model = model_for(kind).to(device)
        model.load_state_dict(torch.load(
            (OUT_DIR if kind == "geometry" else DUAL2D_DIR)
            / f"expert_{kind}" / "seed42.pt",
            map_location=device,
            weights_only=True,
        ))
        graph_map = graph_maps["3d" if kind == "geometry" else "2d"]
        ordered = [graph_map[int(index)] for index in table.source_idx]
        embedding, prediction, indices = encode_expert(
            kind, model, DataLoader(ordered, batch_size=256), device
        )
        if not np.array_equal(indices, arrays["source_idx"]):
            raise ValueError(f"{kind} feature order mismatch")
        arrays[f"{kind}_embedding"] = embedding
        arrays[f"{kind}_prediction"] = prediction
        del model
        torch.cuda.empty_cache()
        print(f"encoded {kind}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        descriptor_rows = list(pool.map(
            router_descriptor_row, table.canonical_smiles, chunksize=500
        ))
    descriptor_frame = pd.DataFrame(descriptor_rows)
    arrays["descriptors"] = descriptor_frame.to_numpy(dtype=np.float32)
    out_path = OUT_DIR / "pilot_expert_features.npz"
    np.savez_compressed(out_path, **arrays)
    manifest = {
        "n": len(table),
        "expert_order": list(EXPERTS),
        "descriptor_columns": descriptor_frame.columns.tolist(),
        "fit_state": "expert checkpoints frozen; descriptors unstandardized",
        "output_sha256": sha256_file(out_path),
    }
    (OUT_DIR / "pilot_expert_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
