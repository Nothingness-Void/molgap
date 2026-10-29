# Decision draft (not closed)

Status: DRAFT for coordinator and user review. Nothing here is executed terminal closure. The
canonical `decision.md` stays "pending" until a reviewer accepts this draft and the role audit
is completed.

## Mechanical and native facts (separate from the science)

Kaggle3 kernel `nvoid912/molgap-k1-sam-100k-s42`, ID 141476354, version 1, COMPLETE. Both arms
completed 40 passes, 31240 optimizer steps and 3998720 presentations, selected weights `live`,
aligned 50000-row development predictions with identical `source_idx` and targets. Strict output
inspection reports `MECHANICALLY_VERIFIED` for both arms with no blockers. Execution retention
bundle retrieved. Runtime qualification and replay readiness are `NOT_EVALUATED` (native proof
tensors were downloaded but not inspected for science). No wall-limit truncation: process wall
5623.5 s (baseline) and 9868.2 s (SAM) against the 18000 s allocation. The preflight overhead gate
for `sam` (ratio <= 2.6) did not stop the run.

## Result against the pre-registered rule (`protocol.md`)

| | Baseline | sam |
|---|---:|---:|
| Development Gap MAE, eV (pass 40, live) | 0.1412945 | 0.1441285 |
| Online train MAE at selected pass, eV | 0.098231 | 0.107213 |
| Dev minus online train, eV | 0.04306 | 0.03692 |
| Process wall, s | 5623.5 | 9868.2 |

gain = baseline minus candidate = -2.834 meV (SAM worse). Paired row bootstrap, 2000 resamples,
seed 42: candidate-minus-baseline error +2.834 meV, 95% interval [+1.936, +3.754] meV, probability
the candidate is better 0.0. The gate needed gain >= 3.0 meV with a lower bound above zero. It
fails by 5.8 meV and the interval excludes zero in the harmful direction. Draft outcome:
`NEGATIVE_UNDER_CONTRACT` for `sam` and `CLOSED` for the same-job baseline reference. No retry,
rho sweep, LookSAM, seed or scale-up follows.

Row uncertainty does not measure training-seed variation. One seed.

## Cost readout (declared, not a gate)

SAM/baseline process-wall ratio 1.755 on T4 (local RTX 5060 step-time ratio was 2.34; the T4 ratio
is lower, so the 18000 s allowance was ample). Device-hour cost per arm is `missing` in the
retrieved records (needs platform allocation accounting).

## Descriptive readouts

- Generalisation gap: the SAM arm has a smaller dev-minus-online-train gap (36.9 vs 43.1 meV) but
  a higher online train error (+9.0 meV) and a higher development error (+2.8 meV). The gap
  shrank because training fit worsened more than development did. Online train MAE is a running
  average under active updates, so this is not a fixed-weight fit.
- Paired curve (candidate minus baseline, meV): SAM leads at passes 1-3 and trails from pass 4 on;
  it is ahead in 6 of 40 evaluations. Window means: passes 1-10 +4.51, 11-20 +3.88, 21-30 +3.74,
  31-40 +3.26, last five +2.95. The deficit shrinks slowly and does not fade out; no late-horizon
  fade flag in the diagnostic.
- Strata by target Gap (candidate minus baseline MAE): below 4 eV +4.15 meV (3725 rows); 4-6 eV
  +2.60 (34837); 6-8 eV +1.40 (10451); at or above 8 eV +21.4 (987). SAM is worse in every
  stratum, most at high Gap. Atom-count strata were not produced by the existing diagnostic; none
  is reported here (post-hoc anyway).
- Row-level: SAM better on 49.0% of rows; median row gain -1.43 meV. Error quantiles shift up
  by roughly 3-6 meV at p50-p95.
- Equal blend of the two arms reaches 0.13681 eV (-4.6 meV vs the better arm), residual
  correlation 0.846. That is complementarity of two separately trained models, not a property
  of the SAM step.

## Declared confound

The perturbed second forward consumes extra CUDA RNG, so later dropout masks differ from the
baseline arm. The comparison is therefore "SAM step plus a different dropout stream", not SAM
alone. Data order is fixed by the CPU sampler and is unaffected. The perturbed pass does not move
BatchNorm running statistics (sandbox-checked on the frozen state). Whether the loss is caused by
sharpness penalty, dropout stream or the larger effective step noise is not identified
(`insufficient_evidence`).

## Attribution

Pre-registered reading: if the dev gap does not shrink, the generalisation story is unsupported.
It did shrink, but only because training error rose faster. At rho 0.05 with LR 4e-4 the SAM step
under-fits (train +9.0 meV) without a development benefit. This is a single rho, single seed result;
it does not show SAM is useless at other rho.

## Items that block closure and need a reviewer

1. `role_use` for `official_validation`, `test_dev`, `test_challenge` is `unknown` in the drafted
   audits (`role_audit_v1/`). The fixed 100K dataset manifest declares those roles unread and the
   kernel mounted only that dataset plus the source package, but the run's role access was not
   observed directly, so `untouched` was not written.
2. Runtime qualification and strict replay readiness are `NOT_EVALUATED`.
3. The authorization (kaggle3 spend, development labels) and the wall/hour ceilings were relayed
   by the coordinator, not received from the user; the 18000 s / 10 T4 h figures were set by the
   coordinator. The reviewer decides whether that provenance is sufficient.
4. Reference digest (6-step CPU state, reference mode) before and after the shared-code edits:
   `439913965b931a23b7c11d7066d3372a1cff03f40615dcbba769d23b548cf77d`, identical; re-verified at
   submission.
