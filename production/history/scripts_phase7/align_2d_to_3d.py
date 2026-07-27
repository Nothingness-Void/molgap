"""
Phase 7: align the 2D embeddings to the 3D molecule set for hybrid fusion.

3D (ETKDG) dropped 371 molecules that 2D kept, so emb_2d[i] != emb_3d[i].
Both lists preserve the source-CSV order, so 3D is an ordered SUBSEQUENCE of 2D.
A two-pointer walk over the y-labels (identical float32 from the same CSV)
recovers, for each 3D molecule, its index in the 2D list — exactly, using order
(not label uniqueness).

Outputs:
  results/phase7/gps_2d_embeddings_aligned.pt   # [N_3d, 192], row-matched to 3D
  results/phase7/align_2d_idx.pt                # int64 [N_3d], 2D indices kept

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/align_2d_to_3d.py
"""
from __future__ import annotations

import torch

from molgap.constants import RESULTS_DIR

PHASE7_DIR = RESULTS_DIR / "phase7"
G2D = PHASE7_DIR / "pyg_2d_graphs_bond_300k.pt"
G3D = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"
EMB_2D = PHASE7_DIR / "gps_2d_embeddings.pt"
OUT_EMB = PHASE7_DIR / "gps_2d_embeddings_aligned.pt"
OUT_IDX = PHASE7_DIR / "align_2d_idx.pt"

ATOL = 1e-4


def main():
    print("Loading 3D graphs ...")
    g3d = torch.load(str(G3D), weights_only=False)
    y3d = torch.stack([g.y.squeeze(0) for g in g3d])
    print(f"  3D: {len(g3d)}")
    del g3d

    print("Loading 2D graphs ...")
    g2d = torch.load(str(G2D), weights_only=False)
    y2d = torch.stack([g.y.squeeze(0) for g in g2d])
    print(f"  2D: {len(g2d)}")
    del g2d

    n2d, n3d = y2d.shape[0], y3d.shape[0]

    # Two-pointer: 3D is an ordered subsequence of 2D.
    keep = []
    i2d = 0
    skipped = 0
    for i3d in range(n3d):
        target = y3d[i3d]
        # advance 2D pointer until labels match (3D skipped those molecules)
        while i2d < n2d and not torch.allclose(y2d[i2d], target, atol=ATOL):
            i2d += 1
            skipped += 1
        if i2d >= n2d:
            print(f"\n[FAIL] ran off end of 2D at 3D idx {i3d} — labels not a clean subsequence")
            return
        keep.append(i2d)
        i2d += 1

    keep = torch.tensor(keep, dtype=torch.long)
    print(f"\nMatched {len(keep)}/{n3d} 3D molecules into 2D")
    print(f"  2D molecules skipped (ETKDG failures): {n2d - n3d} expected, {skipped} walked past")

    if len(keep) != n3d:
        print("[FAIL] match count != 3D count")
        return

    # Verify alignment is exact after re-indexing
    y2d_aligned = y2d[keep]
    max_diff = (y2d_aligned - y3d).abs().max().item()
    print(f"  post-align max label diff: {max_diff:.6f}")
    if max_diff > ATOL:
        print("[FAIL] alignment imperfect")
        return

    # Apply to embeddings
    print("\nLoading 2D embeddings ...")
    emb2d = torch.load(str(EMB_2D), weights_only=False)
    print(f"  emb_2d: {tuple(emb2d.shape)}")
    emb2d_aligned = emb2d[keep]
    print(f"  emb_2d aligned: {tuple(emb2d_aligned.shape)}")

    torch.save(emb2d_aligned, str(OUT_EMB))
    torch.save(keep, str(OUT_IDX))
    print(f"\n[OK] Saved aligned 2D embeddings to {OUT_EMB}")
    print(f"     Saved 2D->3D index map to {OUT_IDX}")


if __name__ == "__main__":
    main()
