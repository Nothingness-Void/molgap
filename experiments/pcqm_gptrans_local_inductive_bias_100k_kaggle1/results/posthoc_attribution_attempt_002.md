# Persistent local-edge saved-artifact attribution

This post-hoc analysis uses only the accepted attempt-002 `rwse16` and
`rwse16_local_edge` development predictions and recorded epoch traces.
Accepted prediction SHA256 values, exact ordered source indices
100000–149999, identical targets, finite values, and sealed protected-role
flags were rechecked. Exact hashes and calculations are in
`posthoc_attribution_attempt_002.json`. The
[terminal decision](../decision_attempt_002.md) retains the frozen verdict.

## Observed pattern

The local-edge increment gains 1.082 meV, below the 3 meV gate. It improves
50.196% of rows and has a +0.266 meV median per-row absolute-error gain.
The candidate predictions shift upward by 2.210 meV; both arms underpredict
on average (RWSE16 −23.266 meV; local-edge −21.056 meV). A same-development
role median-offset exercise reduces the descriptive gain to 0.913 meV.
Unlike centered logits, the small advantage is not explained solely by a
constant output shift. These fitted offsets are not valid evaluation scores.

The EMA-development gain averaged 362.029 meV in epochs 0–9 and 1.405 meV
in epochs 50–59. It declined from 2.236 meV at epoch 49 to 1.082 meV at
epoch 59. The final live-development gain was 1.132 meV, close to the
EMA-development gain, while online training favored the candidate by
5.007 meV. This does not support EMA lag as the main reason the frozen
3 meV gate was missed. It is compatible with an added local path that
improves training fit more than held-out development error; online training
is not a fixed-cohort evaluation and cannot prove overfitting.

The retained training/evaluation intervals total 11,408.949 seconds for
RWSE16 and 16,021.662 seconds for local-edge, a 40.4% larger single-T4
device-time cost for the candidate. In the post-hoc 4–6 eV stratum
(34,837 rows), the candidate is 1.671 meV worse; the smaller 6–8 eV
stratum gains 6.267 meV. Target-based strata are descriptive only.

## Failure-mode disposition

- **Underfitting:** the added path has a better relative online training
  metric, but there is no matched fixed-cohort training evaluation to decide
  whether either arm underfit absolutely.
- **Insufficient exposure:** both arms still improved late, but the *relative*
  gain was shrinking through epoch 59. More exposure cannot be proven
  useless, yet the observed trajectory does not support spending another run
  to chase the 3 meV gate.
- **Module harm:** the path is not uniformly harmful; it produces a small
  paired gain. Under this contract that gain is materially insufficient and
  costs substantially more T4 time.
- **Next cheapest falsifier if reopened:** use a prospectively frozen
  train-only fixed-cohort check to separate the live training fit gap from
  development transfer. Another architecture variation or seed is not
  justified by the present result alone.

No new training, checkpoint inference, protected-role access, or promotion
occurred in this analysis.
