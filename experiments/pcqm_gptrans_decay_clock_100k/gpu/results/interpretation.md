# Interpretation of the isolated G1 decay coefficient

On 2026-10-05 JST, kernel 137044379 version 1 passed independent saved-output
acceptance and the shared RML terminal pipeline. The only scientific
intervention was all-parameter AdamW weight decay 0.05 to 0.01. Model, initial
state, data/roles, EMA999, LR/schedule, FP32, BS128, seed42 and terminal exposure
were verified against the frozen reference. No reference was retrained.

Exact endpoint/row analysis and measurements are owned by
[acceptance](acceptance.json); selected native observations and source hashes
are retained in [trajectory analysis](trajectory_analysis.json).

## Endpoint and trajectory interpretation

- Best EMA MAE increased from 0.1442326291 to 0.1470216409 eV. Candidate minus
  reference was +0.0027890118 eV; the paired-row 95% interval was
  [+0.0018908544, +0.0036815173] eV. This measures row uncertainty, not
  training-seed stability. The prospectively frozen promotion gate failed.
- Candidate EMA development MAE was worse at 59 of 60 aligned epochs. The
  terminal train MAEs were nearly equal (0.097346 versus 0.097143 eV), whereas
  both live and EMA development predictions remained worse. Some middle-stage
  train metrics improved, but no sustained development advantage followed.
- The selected epoch was 50 (zero-based). The final ten-epoch EMA change was
  only -0.00001575 eV; terminal EMA/live differed by 0.00001150 eV. The saved
  observations do not support a remaining EMA-lag explanation or continued
  same-contract training to recover this endpoint.
- Clipping declined from about 91.7% to 0.5%; the candidate weight norm
  stabilized late. No optimizer explosion was observed. The reference has no
  matching norm/clipping diagnostics, so no norm-mediated causal explanation
  was inferred. The evidence is consistent with a regularization trade-off,
  not proof of its internal cause.
- Reference-error quintiles showed tail improvement and easy-row regression,
  but those groups were defined using target errors after observing results.
  They cannot establish chemical specialization, an input-only router, or
  superiority on a fresh role.

## Scope, cost and record qualification

The run completed 46,860 updates / 5,998,080 sample presentations. It used one
visible T4; both allocated T4s were charged in the native cost: 2.22705 wall
hours / 4.45410 allocated T4 hours. No official validation/test role was read.
Local acceptance loaded saved prediction tensors only, not models or graph caches.

The post-run evidence is STRICT_CAUSAL for this single-seed optimizer question;
the [RML closure](../degree_decay001_ema999/results/closure.json) is VALID and
the actual derived pool admits a complete candidate/reference Replay pair.
The immutable terminal label remains INCONCLUSIVE because the owning adapter's
promotion criteria did not pass. It was not rewritten into a scored negative
policy truth merely to increase backtest coverage.

The first local acceptance rejection concerned only the derived cumulative LR
sum: cross-runtime recomputation differed by at most 1.42e-14. The adapter
repair uses a finite, positive recomputation with an eight-ULP bound for that
diagnostic alone. Recorded LR values, native/canonical equality, identities,
hashes, counters and the actual coefficient remain exact checks. Raw evidence,
selection and scientific gates were not altered.

## Bounded conclusion

Close the all-parameter 0.01 coefficient route at 100K; retain the accepted
G1 EMA999 / weight-decay 0.05 reference. Do not resume this candidate, launch a
coefficient grid, or carry its coefficient into 500K. This endpoint does not
disprove a decay/exposure interaction at larger scale, but it rejects the
simple remedy tested here. The separate scale execution profile only measured
feasibility; qualified scale/resume and same-live-trajectory reference ownership
remain prerequisites to any separately authorized 500K scientific job.
