# K1 return-allocation round 3 decision — 2026-09-14

## Question

Does decoupling the atoms that construct K1's molecular slot from those that
receive its update materially improve the frozen PCQM-100K v4 reference?

## Accepted evidence

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-return-allocation-s42`, version 1,
completed both isolated T4 arms. No-inference acceptance passed. Both arms
completed 40 epochs, 31,240 optimizer steps, and 3,998,720 sample
presentations. Data, row order, seed 42, deterministic FP32/no-TF32 mode,
physical batch 128, optimizer, schedule, role access, runtime certificates,
artifact hashes, exact initial K1 function, and parameter counts matched the
frozen v4 contract. Official validation and all test roles remained unread.

| Model | Development Gap MAE | Gain over K1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 3,658,817 | 39 |
| `neural_atom_k1_uniform_return` | 0.1395473480 eV | 0.0018262863 eV | 3,658,817 | 39 |
| `neural_atom_k1_inverse_return` | 0.1420041174 eV | -0.0006304830 eV | 3,658,817 | 39 |

Uniform return had a candidate-minus-reference paired absolute-error bootstrap
95% interval of `[-0.0026759512, -0.0009641610] eV`, entirely favorable.
Inverse-score return had interval
`[-0.0002747395, 0.0015387139] eV`, crossing zero. Both retained more than 96%
of device memory. Resource times are not compared with the P100 reference.

## Attribution

Reusing K1's source-selection weights for return is not necessary. Giving all
valid atoms equal normalized global context produced a credible paired
improvement without adding parameters, changing the slot content, or changing
the initial graph function. Preferentially returning context to atoms with low
source scores was harmful, so low contribution to the molecular summary does
not identify atoms that need stronger global updates.

The uniform gain closely matches the separate no-slot-attention gain
(`0.0017627 eV`). These are two directionally useful simplifications, but each
is below the frozen `0.003 eV` material/run-variation threshold. Their apparent
compatibility does not authorize chaining them: an additive effect has not been
demonstrated, and combining two sub-threshold mechanisms would reopen
micro-variant search after the declared final round.

## Decision

Neither candidate is promoted. Uniform return is retained as directional
design evidence; inverse-score return is closed as scientifically negative.
The authorized three-round K1 sequence is complete. Frozen K1-v4 remains the
server incumbent and the separate desktop full-run handoff remains unchanged.

No extra seed, combination with no-slot attention, shadow read, scale-up,
official-role read, or submission is authorized. A materially different K1
geometry experiment would require a new explicit protocol and authority; it is
not a successor released by this result.

## Evidence pointers

- Launch identity: `results/launch.json`.
- No-inference acceptance: `results/acceptance.json`.
- Retrieved record:
  `platforms/_records/kaggle/training/pcqm_k1_return_allocation_s42_v1/`.
