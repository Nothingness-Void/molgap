# K1 dynamic-query attempt 3 decision

Decision date: 2026-09-13

## Question

Can K1 retain one global atom distribution while conditioning its query on the
current molecule, improving selection without introducing multiple slots or
heads?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-dynamic-query-s42` version 1
completed the frozen 40-epoch contract. No-inference acceptance returned
`accepted=true`. The candidate completed 31,240 optimizer steps and 3,998,720
sample presentations with 3,695,681 parameters. Data, row order, features,
target, optimizer, schedule, role access, runtime certificate, exact nested
initialization, dynamic-query invariants, and artifact hashes passed. Official
validation and all test roles remained unread.

## Result

| Arm | Development Gap MAE | Delta versus K1-v4 | Best epoch | Mean throughput |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 39 | 812.30 graphs/s |
| `neural_atom_k1_dynamic_query` | 0.1423581690 eV | +0.0009845346 eV | 39 | 778.47 graphs/s |

The paired candidate-minus-reference absolute-error bootstrap 95% interval was
`[0.0000387005, 0.0018959908] eV`. It remained entirely unfavorable, although
this was the smallest regression among the three bounded attempts. The
candidate failed the direction and `0.003 eV` material-gain gate. Peak reserved
memory was 620 MiB of 16,269 MiB.

## Attribution

The query conditioners were trainable after the frozen two-step preflight and
the candidate was exactly K1 at initialization. At epoch 39 the candidate had
lower normalized training MAE than K1 (`0.075718702` versus `0.077263069`) but
higher development MAE (`0.142358169` versus `0.141373634`). This repeats the
train-better/development-worse signature of attempts 1 and 2.

Dynamic conditioning was less harmful than four latent groups or four channel
heads, supporting the narrower conclusion that one shared global atom
distribution is the right capacity shape for this 100K Gap task. However, even
a 1% query-conditioning module exploited molecule-specific fitting freedom
without adding transferable information. K1's fixed learned query is therefore
not merely an implementation limitation; under this evidence it functions as a
useful regularizer.

## Final decision

Attempt 3 of 3 is scientifically negative. Close molecule-conditioned K1
queries under this contract; do not rescue them with another context summary,
seed, width, optimizer, schedule, or scale-up. The bounded K1 mechanism sequence
is complete. No candidate qualifies for shadow access, 500K/full training,
official evaluation, or submission. Frozen `neural_atom_k1_v4` remains the sole
server-side architecture candidate eligible for the separate desktop full-run
budget decision.

Large artifacts remain under
`platforms/_records/kaggle/training/pcqm_k1_dynamic_query_s42_v1/`.
