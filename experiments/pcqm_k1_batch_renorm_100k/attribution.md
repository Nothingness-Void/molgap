# Training attribution

Scope: the accepted seed42,100K,40pass mean-correction contrast. This document
interprets terminal_decision.md; it does not alter its negative contract outcome.
Numbers and paired curve are retained in terminal_acceptance/attribution_summary.json
and paired_curve.csv; original shared report remains externally hash-bound.

## What the retained evidence establishes

- Exact source/data/initialization and complete equal exposure are verified;
  both18-module selected counters equal30459. No early termination, missing
  arm or silent extra training forward explains the comparison.
- Both arms are numerically identical for passes1–5 while correction caps are
  identity. This is an observed control for the implemented intervention onset.
- Selected endpoints are both pass39. Mean correction adds1.6455meV error;
  paired row95% interval[0.9067,2.4002]meV is positive. Seed variability is unknown.
- The candidate wins48.724% of rows; its p50/p90/p95/p99 absolute errors are
  higher, though its maximum error is lower. Thus a few largest outliers alone
  do not explain the aggregate loss.
- Posthoc constant-shift correction leaves1.6863meV loss; a global scalar bias
  does not explain it. This uses consumed development labels and is not a gain.
- Additional correction briefly helps during ramp-up, then its mean difference
  is+2.0866meV on passes11–20 and+3.1133meV over the last10. Last5:+3.3907meV.
  A late catch-up hypothesis is unsupported by this horizon; both absolute
  curves still improve, so no universal claim of convergence or exposure
  sufficiency is made. Repeating longer training is not authorized.
- Final-pass regressions from selected are1.0458meV(scale),4.3002meV(affine).
  This is compatible with evaluation-state sensitivity, not proof of buffer
  causality; the selected-weight Clean-BN diagnostic is its separate discriminator.
- Online train loss is an average over changing training weights/modes. It is
  not a matched fixed-weight train MAE and cannot establish over/underfitting.
- Process train windows5439.13s and5432.48s are similar; total observed allocation
  is3.09835 nativeT4hours. This comparison offers no meaningful speed benefit.

## Remaining discriminator and decision

Apply the predeclared same training-prefix Clean-BN diagnostic to both selected
Live models, preserving all weights and original outputs. If the gap collapses,
that supports buffer sensitivity; if it persists, learned-function differences
remain plausible. Neither result uniquely identifies training-time causation.
This is not BRN-versus-BN, nor a failure verdict for every modern normalization.
The local diagnostic and terminal routing must finish before another module.
