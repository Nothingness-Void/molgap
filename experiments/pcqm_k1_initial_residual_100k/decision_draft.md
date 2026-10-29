# Decision draft (not closed)

Status: DRAFT for coordinator and user review. Nothing here is executed terminal closure. The
canonical `decision.md` stays "pending" until a reviewer accepts this draft and the role audit
is completed.

## Mechanical and native facts (separate from the science)

Kaggle3 kernel `nvoid912/molgap-k1-initial-residual-100k-s42`, ID 141309825, version 1, COMPLETE.
Both arms completed 40 passes, 31240 optimizer steps and 3998720 presentations, selected weights
`live`, aligned 50000-row finite development predictions with identical `source_idx` and targets.
Output inspection reports `MECHANICALLY_VERIFIED` for both arms with no blockers. Execution
retention bundle verified and SEALED. Native proof tensors were not re-inspected (`--strict` was not
run; strict retrieval of `native_proofs.zip` has not been done), so runtime qualification and
replay readiness are `NOT_EVALUATED`.

## Result against the pre-registered rule (`protocol.md`)

| | Baseline | initial_residual |
|---|---:|---:|
| Development Gap MAE, eV (epoch 40, live) | 0.1412945 | 0.1420021 |
| Online train MAE at selected pass, eV | 0.098231 | 0.098309 |
| Dev minus online train, eV | 0.04306 | 0.04369 |
| Process wall, s | 5665.8 | 5625.6 |

gain = baseline minus candidate = -0.708 meV (candidate worse). Paired row bootstrap, 2000
resamples, seed 42: candidate-minus-baseline error +0.708 meV, 95% interval [-0.170, +1.551] meV,
probability the candidate is better 0.0515. The gate needed gain >= 3.0 meV with a lower bound
above zero. It fails by 3.7 meV. Draft outcome: `NEGATIVE_UNDER_CONTRACT` for the candidate and
`CLOSED` for the same-job baseline reference. No retry, seed, 500K or adoption follows.

Row uncertainty does not measure training-seed variation. One seed.

## Mechanism readout (descriptive)

Selected candidate gates, L2 norm per 192-vector, layers 2-9: 2.05, 1.62, 1.07, 1.05, 1.09, 1.07,
1.10, 1.15; max absolute value 0.21-0.32. Every norm is far above the 0.05 "unused" threshold, so the
h0 path was used. Layer 2 and 3 gates have clearly positive mean (+0.13, +0.09); layers 6-9 are near
zero mean. Branch of the pre-registration reached: gates grew, gain is below the gate.

## Attribution (insufficient evidence for a cause)

- Paired curve: the candidate is ahead in 18 of 40 evaluations. Mean paired delta is -1.78 meV for
  passes 1-10 and -2.74 meV for passes 11-20 (candidate better), then +0.83 meV over the last 10 and
  +0.75 meV over the last 5 (candidate worse). This is a late-horizon fade, descriptive only.
- Strata by target Gap: below 4 eV the candidate is better (0.2048 to 0.1998, 3725 rows); 4-6 eV
  0.1273 vs 0.1282 (34837 rows); 6-8 eV 0.1581 vs 0.1583; at or above 8 eV worse, 0.2167 vs 0.2379
  (987 rows). Post-hoc, not a router.
- Train fit did not improve (online train 0.098231 vs 0.098309) while the dev gap widened by 0.6 meV,
  so the extra input access did not lower training error and does not support an "information access
  limits accuracy" reading. The online train metric is a running average under active updates, so it
  is not a fixed-weight fit; do not infer overfitting from it.
- Equal blend of the two arms reaches 0.13613 eV (-5.2 meV vs the better arm) with residual
  correlation 0.863. That is output complementarity of two separately trained models, not a
  property of the addon; it needs the same caution as the older fusion results.
- Not identified: training stochasticity (one seed), whether the late fade is schedule, selection or
  mechanism, and any transfer to 500K. Nearest earlier result (RWSE refresh, 1.0 meV at 100K then a
  1.1 meV regression on the fixed 500K rows) points the same way but is a different intervention.

## Items that block closure and need a reviewer

1. `role_use` for `official_validation`, `test_dev`, `test_challenge` is `unknown` in the drafted
   audits (`role_audit_v1/`). The fixed 100K dataset manifest declares those roles unread, and the
   kernel mounted only that dataset plus the source package, but I did not observe the run's role
   access directly, so I did not write `untouched`.
2. Native proof inspection (`--strict`) and the strict comparison pin are not done.
3. The relayed authorization (coordinator message, not the user directly) is recorded in
   `experiment.json`; the reviewer decides whether that provenance is sufficient.

## Reviewer notes (2026-10-30, Opus decision under the user's delegation)

- Protected roles stay `unknown` and this run stays a staged draft. Closure accepts only
  observed role states, and the rules have no "inferred from mounted inputs" level. The
  dataset manifest declares only train and development roles and the kernel mounted only that
  dataset plus the source package, but the kernel had internet access, so these are inferences
  and are not role evidence. Runs submitted after the data-file read record is merged can close
  mechanically; this run predates it.
- Authorization provenance: the user directed these real Kaggle runs in conversation on the
  accounts in their key folder and delegated the remaining closure decisions to Opus on
  2026-10-30, with the condition that the experiment contract is not changed sweepingly.
  Use of the 50K development labels for metrics is the ordinary train/development role use. The
  `experiment.json` authority text is unchanged because it is part of the frozen input.
- Outcome word stays `NEGATIVE_UNDER_CONTRACT` (draft). No merge to molgap-desktop now; when
  the run is eventually closed, the decision and evidence go to RML and the rejected
  implementation is archived (docs/operations/BRANCHES.md).
