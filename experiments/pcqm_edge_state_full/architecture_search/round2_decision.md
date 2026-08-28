# Round 2 Field-Aware Stem Decision

On 2026-08-27, both seed-42 field-aware candidates completed the fixed
official-train-only 100K/10K development protocol. All expected artifacts,
10,000 unique source indices, finite predictions, byte counts, and SHA256
values passed local acceptance. Metrics were independently recomputed from the
retrieved predictions.

| Candidate | Overall MAE | Delta vs control | Radical delta | Non-radical delta | Decision |
|---|---:|---:|---:|---:|---|
| `fieldconcat16` | 0.151517 eV | +0.001455 eV | +0.010850 eV | +0.000678 eV | Reject |
| `fieldconcat32` | 0.150517 eV | +0.000455 eV | +0.003071 eV | +0.000239 eV | Reject |

Neither candidate met the promotion gate. Replacing the summed OGB stem with
a concatenate-and-project stem was therefore closed, and seeds 43 and 44 were
not opened.

The repeated radical-only regression justified one final narrower screen. It
retains the accepted summed stem and GPS body, then adds a graph-level context
computed only from nonzero radical-electron categories. This path is exactly
zero for closed-shell molecules. Its fixed protocol is in
`round3_protocol.md`.
