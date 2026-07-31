# DFT Versus ML Cost

This directory answers one presentation question: how long does one molecule take
by DFT, and how long by the shipped model?

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\compare_dft_vs_ml_cost.py
```

Both sides use the **same ten commercial OLED molecules**. The DFT side is parsed
from the retained Phase 5 Gaussian 16 logs and nothing is recomputed; the ML side
is timed live. A speedup measured across different molecules would not be a
speedup, so the shared molecule set is the whole design of this record.

Results and per-molecule detail: `dft_vs_ml_cost.md` and `dft_vs_ml_cost.json`.

## Which DFT number to quote

The record reports three DFT scopes because they are not interchangeable:

| Scope | What it is | Use it when |
|---|---|---|
| Full `opt freq` | What Phase 5 actually ran: optimize the geometry, then frequencies | Claiming "what a chemist actually runs for one new molecule" |
| Geometry optimization | Optimization alone, no frequency job | Comparing against a geometry-only workflow |
| One geometry step | A single SCF plus gradient, the cheapest honest unit here | Comparing against the single-point labels the model was trained on |

The model was trained on PubChemQC **single-point** B3LYP/6-31G* labels. So the
geometry-step row is the fair per-label comparison, and the `opt freq` row is the
fair end-to-end workflow comparison. Both are real; quoting one while describing
the other is not.

The two scopes differ by roughly 30x, so this choice changes the headline number
materially.

## Caveats that belong on the slide

- The DFT jobs ran on 8 or 16 shared-memory cores, the ML timings on one RTX
  5060. This is a practical wall-clock comparison, not equal-hardware.
- Batched ML throughput is much better than a single call: fixed per-call
  overhead dominates one molecule. Quote the batched number for database-scale
  claims and the single-call number for interactive use.
- DFT cost grows steeply with molecule size; the measured range here is already
  7.5 to 42 minutes across ten molecules of 32 to 80 atoms. The ML cost is
  effectively flat over the same range.
- The ten-molecule accuracy table in the record is an illustrative agreement
  check against a different geometry protocol, not the accepted accuracy
  evidence. Accepted metrics live in `../track_a_final_decision.md`.
- Two of the ten molecules (DPEPO, CzSi) contain P and Si, outside the trained
  CHONSFCl set. They are flagged in the JSON.

## What the comparison does not claim

The model reproduces B3LYP/6-31G* Kohn-Sham eigenvalues, not experiment and not
GW. It cannot produce an optimized geometry, a frequency, a thermochemistry
correction, or anything else the DFT job also yields. The honest framing is that
it replaces one specific expensive lookup, not the DFT calculation as a whole.
