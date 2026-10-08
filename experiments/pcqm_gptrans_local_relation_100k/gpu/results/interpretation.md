# Local relation-flow dual: terminal interpretation — 2026-10-09 JST

## Decision and comparison

On 2026-10-09 JST, both physical-v1 workers passed independent saved-output
acceptance. Preserve A as a below-gate directional observation; close the exact
B intervention without promotion. Neither arm qualified for500K, additional
seeds, a combination or a successor. The original local-bond reference remains
the scale-qualified parent; its separate500K result was not retested here.

| Endpoint, selected EMA | Parameters | Gap MAE, eV | Reference gain, eV | Selected pass | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|
| Frozen local-bond parent | 5,871,201 | 0.1423582275 | — | Frozen reference | Separate execution |
| A: connected valid-pair transition | 5,888,225 | 0.1409914171 | +0.0013668103 | 48 | 242.48 |
| B: same-block true-bond return | 5,896,161 | 0.1435507443 | -0.0011925169 | 49 | 268.71 |

Pass is one-based; native best_epoch is zero-based47/48. MAE was recomputed in
float64 from retained predictions, not copied from rounded console output.
Both independently selected checkpoints were compared with the immutable parent
on the same50,000 internal-development molecules/targets. No baseline retrained.

The prospectively frozen nomination rule required gain>0.003eV and a positive
paired-row interval. A passed the sign criterion but not magnitude. B failed
both. Gain CI95: A[+0.0004888324,+0.0022277446]eV;
B[-0.0022261135,-0.0002263802]eV. These are resampled molecules, **not** training
seed uncertainty. Reused development selection and one seed limit both claims.
The scientific contract or nomination threshold was not changed after results.

## What the observed trajectories distinguish

A led the parent at all60 aligned EMA observations. Gains at passes10/20/30/
40/50/60 were0.001143/0.001571/0.001233/0.001158/0.001790/0.001397eV. Thus its
small benefit was not merely an isolated best-checkpoint fluctuation. All four
return branches, including the last insertion, had positive finite observed
gradient norms at every epoch; the full range was0.0000894–0.0113261. This
addresses the older disconnected-final-branch defect, but does not isolate
virtual-row processing from independent pair nonlinearity: A changed their
scope together on a different, accepted local parent.

B led at only3 of60 matched observations and was worse at all six decadal
observations. All12 return branches had positive finite observed gradients
(0.0000579–0.0364199). The final branch was therefore not simply dead. At pass60,
training MAE was0.0746650 versus the parent's0.0749370, but EMA development MAE
was0.1439281 versus0.1427726. Extra chemical feedback functioned without yielding
commensurate generalization. Redundant/perturbative pair feedback is consistent
with this evidence; neither a specific overfitting cause nor an optimal update
strength was causally identified. No cap/width/seed retry is justified here.

Both EMA endpoints worsened over the last ten passes by about0.00026eV. Final
live/EMA development gaps were only0.0000242(A) and0.0000408(B)eV. The observed
late plateau does not support an unapproved continuation merely to wait for EMA.

Per-molecule wins were50.678%(A) and50.220%(B); count alone would misclassify B.
Post-hoc parent-error quintiles showed losses among easy-parent cases and gains
in its high-error tail for both. Those target-derived groups are diagnostics,
not chemical specialization, deployable routing or proof of an ensemble benefit.
Their improvement/regression balance, not the win count, separates A from B.

## Resources and acceptance boundaries

One T4x2 notebook, two independent isolated workers:4.51457wall hours and
9.02914allocated T4-hours, below the7wall/14allocated cap. A added0.290% parameters;
B added0.425%. B's mean epoch was10.82% longer than A's in this concurrent run;
the intentional second local-message evaluation is real computation despite
the small parameter increase. Historical parent timing is not a matched speed
control. Preflight throughput was475.14/419.40graphs/s and peak reserved memory
about1.18/1.16GiB respectively. A single preflight utilization snapshot is not
average training utilization.

The native validator bound actual kernel137648190/v1, frozen source/spec,
scratch initialization, scientific/runtime/data identities, exact50K row and
target hashes, predictions/selected weights/last resumable state, six independent
ten-pass checkpoint chunks per arm and all60 native/canonical observations.
Both completed46,860 optimizer steps/5,998,080 presentations. Native returned-
update RMS, gradients and weight norms were retained as hash-bound diagnostic
artifacts, separately from the sole canonical replay trace.

No local training or model inference was executed during acceptance. No official
validation, test-dev or challenge role was accessed. Local storage retains the
large payloads/checkpoints; Git retains compact evidence and hashes.

## RML and custody closure

Each arm was independently finalized from its original prospective plan and
passed STRICT_CAUSAL readiness plus actual candidate/reference pool admission:
capability=complete,60 observations, no exclusions. See the
[actual admission proof](replay_pair_proof.json) and
[independent acceptance](acceptance.json). The accepted reference trajectory was
`TC-gptrans-g1-bond-local-100k-s42`; there was no replacement reference or second
baseline run.

The unchanged native adapter conservatively labeled both failed nominations
INCONCLUSIVE. Consequently each trace is admitted, but terminal_label=null and
winner/promotion policy scoring is unavailable. **Complete trace replay admission
is not a fabricated positive/negative policy truth or evidence of promotion.**
The metrics, confidence intervals and per-arm stop decision remain explicit.

The existing Luna monitor delivered one terminal event, then paused. A closed
that claimed event and exact binding without creating a successor or another
monitor. An additional scientific action requires separate authority.

Acceptance/RML plumbing was reused. The only post-run adapter repair binds the
already-produced native mechanism diagnostics to evidence; it changes neither
remote model code, optimizer/selection semantics nor any frozen scientific field.
Its focused static/tensor/retention regressions passed15 tests. Whole-corpus
derived snapshots also contain pre-existing unrelated reconciliations; they are
not bundled into this experiment's classified commits.
