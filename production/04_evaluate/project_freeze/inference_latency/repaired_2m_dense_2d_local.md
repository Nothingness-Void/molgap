# Local Inference Latency - `repaired_2m_dense_2d`

This measures warm end-to-end inference for new SMILES, including graph construction. It is not a precomputed-catalog lookup benchmark.

- Model: `repaired_2m_dense_2d`
- Device: `cuda`
- Model load: `4.467 s` (excluded from warm timings)
- Timed repeats per batch: `3`
- Encoder passes per molecule: `3`

| Inputs | Median batch s | P95 batch s | Median ms/mol | Molecules/s | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0461 | 0.0484 | 46.06 | 21.71 | 47.0 |
| 16 | 0.0511 | 0.0514 | 3.19 | 313.00 | 48.3 |
| 64 | 0.0561 | 0.0578 | 0.88 | 1139.93 | 52.4 |

This preset builds no ETKDG conformer, so its cost is 2D graph construction plus its GPS encoder passes.
