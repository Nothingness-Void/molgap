# Local Inference Latency - `repaired_2m_equal_2d`

This measures warm end-to-end inference for new SMILES, including graph construction. It is not a precomputed-catalog lookup benchmark.

- Model: `repaired_2m_equal_2d`
- Device: `cuda`
- Model load: `4.190 s` (excluded from warm timings)
- Timed repeats per batch: `3`
- Encoder passes per molecule: `2`

| Inputs | Median batch s | P95 batch s | Median ms/mol | Molecules/s | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0274 | 0.0294 | 27.43 | 36.46 | 34.7 |
| 16 | 0.0285 | 0.0310 | 1.78 | 560.57 | 36.0 |
| 64 | 0.0357 | 0.0424 | 0.56 | 1792.96 | 40.1 |

This preset builds no ETKDG conformer, so its cost is 2D graph construction plus its GPS encoder passes.
