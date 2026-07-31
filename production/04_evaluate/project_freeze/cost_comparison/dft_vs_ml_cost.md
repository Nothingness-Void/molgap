# DFT Versus ML Prediction Cost

Same ten commercial OLED molecules on both sides. The DFT numbers are parsed from retained Gaussian 16 logs; nothing was recomputed.

- DFT: `B3LYP/6-31G(d) opt freq`, Gaussian 16, 8-16 shared-memory cores per job
- ML: `cuda`, warm model, 5 timed repeats
- Molecules: 10

## DFT cost per molecule

| Scope | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| Full `opt freq`, wall clock | 23.15 min | 25.16 min | 7.48 min | 41.96 min |
| Geometry optimization, wall clock | 11.37 min | 9.91 min | 2.69 min | 18.42 min |
| One geometry step, wall clock | 46.80 s | 51.30 s | 26.87 s | 78.55 s |
| Full `opt freq`, core-hours | 5.64 core-h | 5.71 core-h | 0.96 core-h | 10.76 core-h |

## ML cost per molecule

| Preset | Passes | Single call | Batch of 10 | Batch of 1,000 |
|---|---:|---:|---:|---:|
| `repaired_2m_dense_2d` | 3 | 41.1 ms | 4.34 ms/mol | 0.75 ms/mol |
| `repaired_2m_equal_2d` | 2 | 21.6 ms | 2.63 ms/mol | 0.60 ms/mol |

## Speedup

| Preset | vs full `opt freq` | vs one geometry step |
|---|---:|---:|
| `repaired_2m_dense_2d` (batched) | 1,842,238x | 62,051x |
| `repaired_2m_equal_2d` (batched) | 2,318,967x | 78,109x |

The model was trained on PubChemQC single-point B3LYP/6-31G* labels, so the geometry-step column is the honest per-label comparison and the `opt freq` column is the honest "what a chemist actually runs" comparison. Quote whichever you mean, and say which one it is.

## Accuracy on these same molecules

Mean absolute error against the Gaussian reference, in eV:

| Model | HOMO | LUMO | Gap | Average |
|---|---:|---:|---:|---:|
| Phase 5 SchNet (historical) | 0.216 | 0.196 | 0.352 | 0.255 |
| repaired-2M equal | 0.077 | 0.100 | 0.139 | 0.105 |
| repaired-2M dense | 0.075 | 0.099 | 0.126 | 0.100 |

This is a ten-molecule spot check against a different DFT protocol (`B3LYP/6-31G(d)` opt+freq geometries, not PubChemQC PM6 geometries), so it is an illustrative agreement check, not the accepted accuracy evidence. Accepted metrics are in `../track_a_final_decision.md`.

Two of the ten contain elements outside the trained CHONSFCl set and are flagged in the JSON record.
