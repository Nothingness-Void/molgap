"""
Phase 7: verify 2D and 3D graph lists describe the SAME molecules in the SAME
order, so emb_2d[i] corresponds to emb_3d[i] for hybrid fusion.

Both graph lists are built from the same CSV with identical filtering, but each
drops build-failures without leaving a gap. If counts differ (or labels don't
match row-by-row) the index alignment is broken and fusion would pair wrong
molecules.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/check_alignment.py
"""
from __future__ import annotations

import torch

from molgap.constants import RESULTS_DIR

PHASE7_DIR = RESULTS_DIR / "phase7"
G2D = PHASE7_DIR / "pyg_2d_graphs_bond_300k.pt"
G3D = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"


def main():
    print("Loading 3D graphs ...")
    g3d = torch.load(str(G3D), weights_only=False)
    print(f"  3D graphs: {len(g3d)}")

    print("Loading 2D graphs ...")
    g2d = torch.load(str(G2D), weights_only=False)
    print(f"  2D graphs: {len(g2d)}")

    if len(g2d) != len(g3d):
        print(f"\n[MISMATCH] counts differ: 2D={len(g2d)} vs 3D={len(g3d)}")
        print("Index alignment is BROKEN — fusion needs label-based re-matching.")
        return

    # Same count — verify labels match row-by-row
    y2d = torch.stack([g.y.squeeze(0) for g in g2d])
    y3d = torch.stack([g.y.squeeze(0) for g in g3d])

    max_diff = (y2d - y3d).abs().max().item()
    n_mismatch = (~torch.isclose(y2d, y3d, atol=1e-4)).any(dim=1).sum().item()

    print(f"\nLabel comparison (row-by-row):")
    print(f"  max abs diff: {max_diff:.6f}")
    print(f"  rows with mismatch (atol=1e-4): {n_mismatch}/{len(g2d)}")

    if n_mismatch == 0:
        print("\n[OK] 2D and 3D graphs are ALIGNED — index pairing is valid.")
    else:
        print(f"\n[MISMATCH] {n_mismatch} rows differ — alignment BROKEN despite equal counts.")


if __name__ == "__main__":
    main()
