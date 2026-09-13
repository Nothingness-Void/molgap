# K1 readout and selector decision — 2026-09-13

## Question

Can either a RepSet final-node readout or a selector shared across K1 exchange
layers materially improve the frozen PCQM-100K v4 K1 reference?

## Accepted evidence

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-readout-selector-s42`, version 1,
completed both isolated T4 arms. No-inference acceptance passed. The fixed
dataset, row order, seed 42, deterministic FP32/no-TF32 mode, physical batch
128, optimizer, schedule, 40-epoch exposure, development role, and sealed-role
flags match the immutable K1-v4 reference contract.

| Model | Development Gap MAE | Difference from K1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 3,658,817 | 39 |
| `neural_atom_k1_repset_readout` | 0.1421080679 eV | +0.0007344335 eV | 3,683,985 | 39 |
| `neural_atom_k1_tied_selector` | 0.1418427527 eV | +0.0004691184 eV | 3,622,401 | 38 |

Candidate-minus-reference absolute-error bootstrap intervals were
`[-0.0001722729, 0.0016259335] eV` for RepSet and
`[-0.0004616975, 0.0013987419] eV` for the tied selector. Both cross zero and
neither reaches the frozen `0.003 eV` material-gain gate. All 40 epochs
completed and the best epochs were at the end, so this is not a truncated-run
failure.

## Decision

Both round-1 candidates are scientifically closed. They receive no additional
seed, shadow read, scale-up, official-role read, or submission authority. The
frozen K1-v4 reference remains unchanged.

## Attribution and round-2 implication

The RepSet branch added final-set capacity but did not improve transfer. The
tied selector removed layer-specific atom addressing and stayed closer to K1,
but still produced no reliable gain. Together with the earlier multi-slot,
multi-head, dynamic-query, gate, and relation-slot failures, this strengthens
the conclusion that additional allocation/readout freedom is not the missing
capacity.

K1 has exactly one active latent slot. Self-attention over a length-one slot
set has a constant attention weight, so its query/key projections cannot model
slot-to-slot interaction. Round 2 may therefore test only whether collapsing
or removing this redundant single-slot attention processor improves the same
K1 information flow. It must retain the exact v4 contract and immutable
reference.

## Evidence pointers

- Retrieved record:
  `platforms/_records/kaggle/training/pcqm_k1_readout_selector_s42_v1/`
- No-inference acceptance: `round1_acceptance.json` in that record.
- Launch identity: `results/launch.json`.
- Monitor incident and repair: `results/monitor_incident.md`.

