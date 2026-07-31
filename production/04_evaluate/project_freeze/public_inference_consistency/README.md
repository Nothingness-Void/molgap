# Public Inference Consistency

`PACKAGE-A` requires a **tested** inference loader, not just a loader. This
directory answers one question: does the public `molgap.inference` path return
the same accuracy that the accepted Track A external evaluation recorded?

The check replays the accepted 1,973-row external prediction table through the
public API and compares average MAE per scope. It reads no sealed set, trains
nothing, and selects nothing.

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\verify_repaired_2m_public_inference.py
```

## Result

Both presets pass. Public minus accepted average MAE, in eV:

| Preset | All (1,973) | OOD (998) | P8-hard (975) | Max per-row difference |
|---|---:|---:|---:|---:|
| `repaired_2m_dense_2d` | `+1.40e-06` | `+7.01e-06` | `-4.34e-06` | `8.54e-03` |
| `repaired_2m_equal_2d` | `+1.06e-05` | `+2.36e-05` | `-2.79e-06` | `5.29e-03` |

The gate is `1e-4 eV` on scope average MAE, two orders of magnitude below the
`0.005-0.006 eV` improvement the Track A decision claims over routed-v4.

Per-row differences are larger than the aggregate ones because the accepted
table was produced under CUDA autocast while the public path runs fp32. That is
a numeric-precision difference, not a different model: the same checkpoints and
the same three-seed gate ensemble are loaded, verified by SHA256 in
`repaired_2m_public_inference.json`.

## What this does not establish

This confirms the public path reproduces an already accepted evaluation. It is
not a new accuracy measurement and grants no additional claim. Accuracy evidence
and its boundaries stay in `../track_a_final_decision.md`.
