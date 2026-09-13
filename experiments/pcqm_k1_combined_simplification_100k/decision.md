# K1 combined-simplification decision — 2026-09-14

## Question

Do removal of K1's degenerate length-one slot attention and uniform normalized
return combine into a material improvement over the frozen K1-v4 reference?

## Accepted evidence

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-k1-combined-s42`, version 3,
completed the candidate. No-inference acceptance passed after repairing an
acceptance-only type mismatch: the shared loader requires a mode-to-parameter
mapping, while the first acceptance invocation passed a scalar. The remote
training artifacts were not changed or re-executed.

The candidate completed 40 epochs, 31,240 optimizer steps, and 3,998,720
sample presentations. Data, row order, seed 42, deterministic FP32/no-TF32,
physical batch 128, optimizer, schedule, loss, role access, source identity,
runtime certificate, artifact hashes, and exact initial K1 function satisfied
the frozen V4 contract. Official validation and all test roles remained
unread.

| Model | Development Gap MAE | Gain over K1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 3,658,817 | 39 |
| combined simplification | 0.1392112225 eV | 0.0021624118 eV | 3,608,897 | 35 |

The paired candidate-minus-reference absolute-error bootstrap 95% interval was
`[-0.0030205362, -0.0012869453] eV`, entirely favorable, and 51.068% of rows
improved. The candidate retained 96.04% device-memory reserve.

## Attribution

The two simplifications are partially additive. Their combination beat the
no-slot-attention parent by `0.0003997087 eV` and the uniform-return parent by
`0.0003361255 eV`. This supports the mechanism-level conclusion that the
length-one processor and asymmetric return introduce separate small costs.

The combined gain nevertheless remains below the predeclared `0.003 eV`
material/run-variation threshold. A favorable paired interval establishes a
directional effect on this reused development role; it does not override the
minimum-gain gate or justify another seed and scale-up.

## Decision

The combination is scientifically directional but does not pass promotion.
It is closed without another seed, shadow read, scale-up, official-role read,
or submission. Frozen K1-v4 remains the server incumbent and its separate
desktop full-run handoff is unchanged. This interaction result does not release
another K1 micro-variant.

## Evidence pointers

- No-inference acceptance: `results/acceptance.json`.
- Retrieved record:
  `platforms/_records/kaggle/training/pcqm_k1_combined_s42_v3/`.

