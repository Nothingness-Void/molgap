# Track A Final Decision

## Decision

Track A research was frozen on 2026-07-30. The selected general B3LYP model is
the repaired-2M three-GPS dense pure-2D ensemble.

The repaired-2M GPS7/GPS9 equal ensemble is retained as the lower-cost preset.
The dual-SchNet residual path is rejected and is not part of either frozen
model.

## External Evidence

All methods were compared on the same 1,973 common molecules with valid
ETKDGv3+MMFF200 coordinates. Average MAE is in eV.

| Model | All | OOD | P8-hard | Encoder passes |
|---|---:|---:|---:|---:|
| Routed-v4 500K | 0.103580 | 0.112721 | 0.094222 | Existing production path |
| Repaired-2M GPS7/GPS9 equal | 0.098467 | 0.108798 | **0.087892** | 2 GPS |
| Repaired-2M three-GPS dense | **0.097638** | **0.106655** | 0.088407 | 3 GPS |

Both repaired-2M candidates improved the routed-v4 average MAE in every scope,
and every paired average-MAE 95% interval was below zero. The dense candidate
improved all-set Gap MAE by `0.008348 eV`; the equal candidate improved P8-hard
Gap MAE by `0.010525 eV`.

## Rejected 3D Path

Adding the two accepted SchNet branches increased average MAE against the
corresponding frozen 2D identity by:

- `+0.023251 eV` for GPS7/GPS9 equal;
- `+0.024239 eV` for three-GPS dense.

The SchNet checkpoints remain reproducibility assets, but no Track A inference
path may call these rejected residual heads.

## Delivery Boundary

This decision freezes the scientific model identity. The registered public
default remains routed-v4 until the repaired-2M checkpoints are in the local
model inventory and a tested inference loader, latency record, and invalid/OOD
SMILES smoke test are complete. This remaining work is packaging, not model
research.

No new Track A architecture, dataset, Router, MoE, seed, or fusion experiment is
authorized before the presentation.

Source evidence:

- `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/decision.md`
- `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/acceptance.json`

