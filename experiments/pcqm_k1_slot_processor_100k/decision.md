# K1 single-slot processor decision — 2026-09-13

## Question

Does collapsing or removing K1's mathematically degenerate length-one slot
self-attention materially improve the frozen PCQM-100K v4 K1 reference?

## Accepted evidence

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-slot-processor-s42`, version 1,
completed both isolated T4 arms. No-inference acceptance passed. The fixed
dataset, row order, seed 42, deterministic FP32/no-TF32 mode, physical batch
128, optimizer, schedule, 40-epoch exposure, development role, and sealed-role
flags match the immutable K1-v4 reference contract.

| Model | Development Gap MAE | Gain over K1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 3,658,817 | 39 |
| `neural_atom_k1_collapsed_mha` | 0.1408551633 eV | 0.0005184710 eV | 3,633,857 | 39 |
| `neural_atom_k1_no_slot_attention` | 0.1396109313 eV | 0.0017627031 eV | 3,608,897 | 39 |

Candidate-minus-reference absolute-error bootstrap intervals were
`[-0.0014302194, 0.0003692285] eV` for collapsed MHA and
`[-0.0025974751, -0.0009116281] eV` for no slot attention. Removing the
attention residual therefore produced a directionally credible paired
improvement, but it remained below the frozen `0.003 eV` material-gain and
run-variation gate. Both arms completed all 40 epochs, and their best epoch was
39, so the comparison was not decided by truncation.

## Decision

Neither round-2 candidate is promoted. No extra seed, shadow read, scale-up,
official-role read, or submission is authorized. The frozen K1-v4 reference
remains unchanged.

## Attribution

The length-one attention processor was unnecessary: its removal reduced
parameters by 49,920, shortened the mean epoch from 144.23 to 142.71 seconds
relative to the collapsed arm, and improved paired development errors. The
effect is nevertheless too small to distinguish from the accepted cross-run
variation budget. This closes further query/key/value or one-token attention
micro-variants. Any final authorized round must address a different K1
information-flow bottleneck rather than continue simplifying this processor.

## Evidence pointers

- Retrieved record:
  `platforms/_records/kaggle/training/pcqm_k1_slot_processor_s42_v1/`
- No-inference acceptance: `results/acceptance.json`.
- Launch identity: `results/launch.json`.
