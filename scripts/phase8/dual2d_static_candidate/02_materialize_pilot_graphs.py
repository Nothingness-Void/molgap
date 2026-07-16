"""Materialize the active dual-2D pilot cache from expansion500k."""

from __future__ import annotations

import json

import pandas as pd
import torch

from molgap.constants import RESULTS_DIR
from molgap.pubchemqc import sha256_file


def subset_cache(source, destination, needed):
    graphs = torch.load(source, weights_only=False)
    selected = [
        graph for graph in graphs if int(graph.source_idx.item()) in needed
    ]
    del graphs
    torch.save(selected, destination)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "graphs": len(selected),
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }


def main() -> None:
    phase8 = RESULTS_DIR / "phase8"
    out_dir = phase8 / "dual2d_static_candidate"
    table = pd.read_parquet(out_dir / "pilot_30k.parquet")
    needed = set(table.source_idx.astype(int))
    records = {"2d": subset_cache(
        phase8 / "pyg_2d_graphs_bond_expansion_500k.pt",
        out_dir / "pilot_30k_graphs_2d.pt",
        needed,
    )}
    (out_dir / "pilot_graph_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
