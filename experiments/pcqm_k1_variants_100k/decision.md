# K1 selective-global PCQM-100K v4 decision

Decision date: 2026-09-12

## Question

Could either molecule-conditioned global-exchange strength (`K1-G`) or a
separate EdgeState-derived relation slot (`K1-R`) materially improve the
frozen one-slot Neural-Atom K1 architecture?

## Acceptance

Kaggle2 reference kernel `kaseichou/molgap-pcqm-k1-v4-reference-s42` version 2
and candidate kernel `kaseichou/molgap-pcqm-k1-v4-candidates-s42` version 2
completed the frozen 40-epoch contract. Joint no-inference acceptance returned
`accepted=true`; all three arms completed 31,240 optimizer steps and 3,998,720
sample presentations. Fixed-data, row-order, feature, target, optimizer,
schedule, role-access, runtime-certificate, and artifact hashes passed. Official
validation, test-dev, and test-challenge remained unread.

The P100 reference and T4 candidate runtimes have separate accepted
certificates. Hardware identity is provenance under comparison policy v4 and
does not change the scientific contract. Runtime and memory values are not used
as cross-accelerator speed claims.

## Result

| Arm | Development Gap MAE | Delta versus K1-v4 | Parameters | Best epoch | Accelerator |
|---|---:|---:|---:|---:|---|
| `neural_atom_k1_v4` | 0.1413736343 eV | reference | 3,658,817 | 39 | P100 |
| `neural_atom_k1_g` | 0.1452639550 eV | +0.0038903207 eV | 3,698,180 | 36 | T4 |
| `neural_atom_k1_r` | 0.1415209919 eV | +0.0001473576 eV | 3,739,841 | 39 | T4 |

For K1-G, the paired candidate-minus-reference absolute-error bootstrap 95%
interval was `[0.0029734144, 0.0048308560] eV`: it was consistently worse and
failed the 0.003 eV material-gain gate in the wrong direction. For K1-R, the
interval was `[-0.0007295501, 0.0010152798] eV`: it was statistically
indistinguishable from the reference and far below the materiality floor.
`selected_candidate` was therefore null.

## Attribution

The added mechanisms were active rather than accidentally frozen. The three
zero-initialized K1-G output layers acquired nonzero weights, and the three
K1-R return projections acquired substantial nonzero norms. K1-G ended with a
slightly worse normalized training error as well as a worse development error,
so molecule-conditioned scaling interfered with the already useful fixed
exchange instead of allocating it better. K1-R achieved a lower normalized
training error than K1-v4 but a slightly worse development score, indicating
extra relation-slot capacity without transferable information gain.

The v4 reference score must not be compared numerically with historical v3 K1
scores as evidence of regression: the role size, tail-batch policy, exposure,
and runtime contract differ. Its purpose is the reusable baseline for this
complete v4 benchmark contract.

## Decision

Both K1-G and K1-R are closed. Do not retry them through another seed, width,
gate range, relation width, placement, optimizer, schedule, or scale-up. Neither
qualifies for shadow access, 500K/full training, official evaluation, or
submission. Frozen K1 remains the sole server-side candidate eligible for the
separate desktop full-run budget decision.

Retrieved large artifacts remain under
`platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/` and
`platforms/_records/kaggle/training/pcqm_k1_v4_candidates_s42_v2/`. The latter
contains `joint_acceptance.json`; model and payload hashes are recorded there.
