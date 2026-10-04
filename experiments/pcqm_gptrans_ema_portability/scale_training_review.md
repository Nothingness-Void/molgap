# Matched G1 EMA scale follow-up — release review, 2026-10-04

This document records a proposed experiment and capability gaps, not a frozen
training contract, prospective run, compute release or submission receipt.

## Evidence-directed question

The [accepted audit](attempt_v4/decision.md) supported testing corrected EMA
under fixed500K optimization. It did not support another path/readout/additive
module. The 100K source intervention matched all live updates, so independent
duplicate encoder training is unnecessary in principle: two EMA filters can
observe a single unchanged live optimizer trajectory. This optimization needs
explicit shared-trajectory semantics and acceptance; it is not two independent
RNG/optimizer arms and must not be described as such.

## Proposed controlled design

- Accepted fixed500K train [0,500000), development [500000,550000); no other role.
- G1 degree-scaled GPTrans, 12x256 node / 32 pair, 5,246,817 parameters.
- Same hash-bound degree-scaled random initialization; no selected-model warm start.
- FP32, TF32 off, physical BS128; identical optimizer/sampler/normalization,
  fixed optimizer-step learning-rate schedule and sample presentation endpoint.
- EMA 0.9999 versus 0.999, updated from the same live states after every step.
  Independent filter state, selected endpoint, saved predictions and trace fields.
- Every evaluation must restore live parameters/mode and preserve RNG states
  so observing an extra filter cannot change the gradient trajectory.
- Atomic resume must bind live model, both EMA states, optimizer, scheduler,
  RNG, sampler cursor and independently retrievable checkpoint chunks.
- Prespecified paired endpoint analysis, live/EMA curves, native allocation
  costs, role history and honest RML publication. No automatic extra seed/full run.

Keep total presentations fixed when studying dataset diversity: approximately
six million presentations is about 12 passes at500K, not 60. A step-normalized
schedule would retain the 100K optimization horizon. This answers an
equal-exposure scale question, not whether a fully converged500K model wins.
Alternatively 60 passes gives five times the presentations and asks a different
question. Neither choice is silently interchangeable with old results. Freeze
one endpoint and a study-specific material gate before running; the audit's
50% portability nomination is not an automatically inherited training gate.

## Actual reuse and unsupported boundaries

`pcqm_gptrans_v4` and its qualified author/RML adapter reject non100K assets,
hard-code60 passes and development indices, and support only one EMA state.
Changing its constants at runtime would bypass published identity/resume checks.
`pcqm_gptrans_prenorm_500k` accepts the real500K cache but exposes only reference
and Pair PreNorm, uses all500K-derived target statistics and lacks G1/dual-EMA
qualified terminal wiring. Calling it with a renamed mode would not run G1.
The shared graph trainer uses a different no-EMA recipe/input ABI and is not
an equivalent replacement. Their old scalar/reference evidence cannot supply
a qualified frozen reference for this new scientific contract.

Reuse their existing EMA update, deterministic optimizer, accepted500K role
loader, hashing/atomic IO, source/receipt workflow and RML trace/terminal core.
Add only an owning scale/filter hook and explicit pending paired-reference
qualification if the shared-trajectory design is approved. Do not fabricate an
accepted reference before it exists, or relax the server release gate.

## Compute-release conditions

1. A reviewed executable scale/filter adapter and focused synthetic/static
   tests qualify the frozen recipe, dual-filter state, resume and role boundaries.
2. The scientific comparison gate explicitly supports a prospectively frozen
   same-trajectory filter control. If it does not, freeze a reference acquisition
   phase honestly; do not claim strict/replay eligibility from a fake bundle.
3. The target transform is frozen before release, with its training-only source
   and exact bytes; it is not recomputed differently for each arm/platform.
4. CPU source/mount/initialization checks pass. GPU optimizer-inclusive preflight
   must fit the separately frozen native budget before encoder training begins.
5. Use only server-owned Kaggle2/3 under reconciled account/data/quota authority;
   do not adopt desktop or borrow IMS/SCNet custody to bypass a budget gap.
6. Bind actual returned kernel/version and source to the existing Luna monitor.

At this review no matched G1 scale training was submitted. The missing capability
was recorded instead of relabeling an unsupported legacy trainer as release-ready.
