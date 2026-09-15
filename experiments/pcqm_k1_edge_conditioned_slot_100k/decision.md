# K1 edge-conditioned slot decision — 2026-09-16

## Question

Can the three sparse K1 slot selectors use the incident persistent real-bond
state as an additional key context, while preserving the one-slot pure-2D K1
contract?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 3
completed all 40 epochs. Saved-artifact acceptance returned
`accepted=true` without constructing or executing a model. The candidate
completed 31,240 optimizer steps and 3,998,720 sample presentations under the
fixed PCQM4Mv2 100K/50K development contract. The source commit was
`29cb2b9a6144b1f3badc70035a16ca6df0843333`; the source archive SHA-256 was
`6e16f4011f860cc4e31615d5a936c410f632e4ae9f55f4fbe98d884a8ae84a65`.

The data, row order, features, target, optimizer, schedule, seed, role access,
and selection identities matched the immutable K1-v4 reference. The runtime
was strict FP32/no-TF32 on one Tesla T4 with physical batch 128. Official
validation, shadow, test-dev, and challenge roles remained unread, and
`model_inference_executed=false`.

## Result

The gain column is frozen-reference development MAE minus candidate
development MAE; positive values are favorable.

| Arm | Parameters | Best epoch | Development Gap MAE | Gain vs K1-v4 | Mean throughput | Peak allocated / reserved | Device memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen `neural_atom_k1_v4` | 3,658,817 | 39 | 0.1413736343 eV | — | 812.30 graphs/s | 629.776 / 672 MiB | 16,269.25 MiB P100 |
| `neural_atom_k1_edge_conditioned_slot` | 3,671,105 | 39 | 0.1410829425 eV | +0.0002906919 eV | 790.15 graphs/s | 543.898 / 586 MiB | 14,911.6875 MiB T4 |

The candidate adds 12,288 parameters (`+0.3358%`). Its memory gate passed
with a 0.960702 reserve fraction. Throughput was 2.73% lower than the
reference and mean epoch time was 3.45 seconds longer; resource pressure was
not the failure mode.

The paired candidate-minus-reference absolute-error bootstrap 95% interval
was `[-0.0012159737, 0.0006187779] eV`. Its upper bound is not below zero, and
the point estimate is about one tenth of the required `0.003 eV` material gain.
The machine gate therefore correctly returned `passed=false` and
`selected_candidate=null` despite mechanical acceptance.

## Scientific attribution

The mechanism was actually exercised. Preflight verified that the candidate
was exactly K1 at initialization, that its three zero-initialized 64-to-64
edge-key projections became trainable after two optimizer steps, and that all
nine persistent edge states were finite and real-bond-only, their update
context was normalized, and they were connected to mass-one slot assignments
at layers 3, 6, and 9. Resume
determinism and the runtime certificate also passed.

The candidate fit the training role more tightly: epoch-39 normalized training
MAE fell from the reference's `0.0772630692` to `0.0762633287`. That extra fit
translated into only `0.0002906919 eV` on the ordered 50,000-row development
role, with a paired interval compatible with no improvement. The evidence
supports a narrow conclusion: incident local edge memory is a learnable input
to the sparse selector, but under this matched 100K direct-Gap contract it does
not provide a stable, material transfer signal. The small favorable point
estimate is insufficient to distinguish useful information from run-level
variation; the modest throughput cost adds no compensating systems benefit.

## Final decision

This seed-42 screen is scientifically negative at the predeclared promotion
gate. Close the edge-conditioned-slot question under this contract. Do not
grant shadow access, official/test evaluation, a seed 43/44 repeat, a scale-up,
or a successor from this result. Frozen `neural_atom_k1_v4` remains unchanged
and remains the only K1 candidate eligible for the separately governed desktop
full-run decision.

The complete downloaded evidence is retained under
`platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v3/`.
The machine acceptance record is
`results/acceptance.json`.
