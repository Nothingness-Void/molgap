# Decision draft (not closed)

Status: DRAFT for coordinator and user review. Nothing here is executed terminal closure. The
canonical `decision.md` stays "pending" until a reviewer accepts this draft and the role audit
is completed.

## Mechanical and native facts (separate from the science)

Kaggle3 kernel `nvoid912/molgap-k1-extremal-readout-100k-s42`, ID 141523685, version 1, COMPLETE.
Both arms completed 40 passes, 31240 optimizer steps and 3998720 presentations, selected weights
`live`, aligned 50000-row finite development predictions with identical `source_idx` and targets.
Strict output inspection reports `MECHANICALLY_VERIFIED` for both arms with no blockers. Native
proof dependencies were downloaded and hash-verified (`DEPENDENCIES_VERIFIED`); runtime
qualification and replay readiness are `NOT_EVALUATED`. Process wall 5626.5 s (baseline) and
5864.4 s (candidate), ratio 1.042, well under the 14400 s allocation.

## Result against the pre-registered rule (`protocol.md`)

| | Baseline | candidate (extremal readout) |
|---|---:|---:|
| Development Gap MAE, eV (live, selected pass) | 0.1412945 (pass 40) | 0.1446276 (pass 39, step 30459) |
| Final-pass MAE, eV | 0.1412945 | 0.1452270 |
| Online train MAE at selected pass, eV | 0.098231 | 0.095054 |
| Dev minus online train, eV | 0.04306 | 0.04957 |
| Process wall, s | 5626.5 | 5864.4 |

gain = baseline minus candidate = -3.333 meV (candidate worse). Paired row bootstrap, 2000
resamples, seed 42: candidate-minus-baseline error +3.333 meV, 95% interval [+2.303, +4.320]
meV, probability the candidate is better 0.0. The gate needed gain >= 3.0 meV with a lower
bound above zero. It fails by 6.3 meV and the interval excludes zero in the harmful direction.
Draft outcome: `NEGATIVE_UNDER_CONTRACT` for the candidate and `CLOSED` for the same-job
baseline reference. Under the pre-registered bet this closes the K1 extremal and second-order
readout family at 100K. No retry, variant, seed or scale-up follows.

Row uncertainty does not measure training-seed variation. One seed.

## Mechanism readout (descriptive)

Selected candidate projection `W` (192 x 576, zero at start), Frobenius norm per statistic block:
max 3.616, min 3.581, std 2.949, total 5.882; largest single entry 0.156 (max block), 0.161
(min), 0.090 (std). The branch was used (it moved far from zero), so "unused path" does not
explain the result. Whether the learned use is helpful or harmful is not identified.

## Attribution (insufficient evidence for a cause)

- Paired curve (candidate minus baseline, meV): window means passes 1-10 -10.41, 11-20 -1.98,
  21-30 +2.40, 31-40 +3.81, last five +3.52. The candidate is ahead in 17 of 40 evaluations. This
  is an early lead that reverses at about pass 4-5 and stays behind from pass 21 on; the
  diagnostic flags `late_horizon_fade`. Descriptive only.
- Train fit did not improve in step with development: online train MAE is 3.2 meV lower for the
  candidate while development MAE is 3.3 meV higher, so the dev-minus-train gap widens from
  43.1 to 49.6 meV. The online train metric is a running average under active updates, not a
  fixed-weight fit; do not read it as overfitting proof.
- Global shift: candidate mean signed error +46.6 meV against +3.9 meV for the baseline. A
  post-hoc constant shift removes 142% of the delta (shift-corrected candidate 0.139867 vs
  baseline 0.141281, -1.41 meV, candidate better). This is a bias of the selected endpoint, not a
  usable result: a shift fitted on development labels is not available at inference, and the
  selected pass is the live best-development one. The cause of the offset is not identified.
- Strata by target Gap (candidate minus baseline MAE): below 4 eV +39.0 meV (3725 rows); 4-6 eV
  +1.7 (34837); 6-8 eV -3.0 (10451); at or above 8 eV -5.6 (987). The loss sits in the low-Gap
  tail. Post-hoc, not a router.
- Row level: candidate better on 48.1% of rows; median row gain -3.3 meV.
- Equal blend of the two arms reaches 0.13517 eV (-4.1 meV vs the better arm) with residual
  correlation 0.829. That is output complementarity of two separately trained models, not a
  property of the addon.
- Frozen-probe comparison: the pre-run probe ceiling for final-layer statistics was +1.25 meV
  (different run, MAE 0.1431, post-hoc on development labels). The trained branch did worse than
  nothing (-3.3 meV), so the probe ceiling did not predict a gain; this is consistent with the
  stated low prior.
- Not identified: training stochasticity (one seed), whether the offset and the late fade are
  caused by the branch, the schedule or the selection, and any transfer to 500K.

## Items that block closure and need a reviewer

1. Resolved 2026-10-30: `role_audit_v2/` records `untouched` for all three protected roles with
   status `structural_untouched` (structural, not observed; see the reviewer notes). `role_audit_v1/`
   stays as the earlier `unknown` draft.
2. Native proof tensors were downloaded and hash-verified but not re-inspected for science, so
   runtime qualification and replay readiness stay `NOT_EVALUATED`.
3. The relayed authorization (coordinator message, not the user directly) is recorded in
   `experiment.json`; the reviewer decides whether that provenance is sufficient.

## Reviewer notes

- Protected roles: structural, not observed. Frozen source 6b2995e13177f249e3d91dc411a5e5f246036b36 loads data only through
  load_training_data (blob 6c6bc503), which selects only train and development. Pinned manifest
  1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d declares only those roles, with all
  protected-read flags False. The kernel mounted only nvoid912/k1-extremal-readout-100k-src and nvoid912/pcqm4mv2-ogb-fixed-100k-v1
  (kernel-metadata sha 62313ff094045b1ca9c4d70d45f61a60409bc51b3d28e82c344fc092f8a7cf2a). `untouched` means the protected roles were not among this run's inputs.
  File opens were not observed, and this draft makes no claim about what the process could reach or
  about network access (the kernel had internet). Recorded in `role_audit_v2/` by
  `_protected_roles_from_frozen_inputs` in `experiment_terminal_staging.py`.
- Authorization provenance: the real Kaggle submission and the use of the 50K development labels
  for metrics were relayed by the coordinating agent, not received directly from the user. The
  `experiment.json` authority text says so and is unchanged because it is part of the frozen
  input. The ceilings (8 T4 hours, 14400 s) were set by the coordinator.
- Outcome word stays `NEGATIVE_UNDER_CONTRACT` (draft) and needs a reviewer to confirm. No merge
  to molgap-desktop now; when the run is eventually closed, the decision and evidence go to RML
  and the rejected implementation is archived (docs/operations/BRANCHES.md).
