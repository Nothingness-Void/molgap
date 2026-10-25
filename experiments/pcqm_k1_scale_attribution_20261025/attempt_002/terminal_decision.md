# Retained-tail diagnostic terminal decision — 2026-10-25

Accept this fixed CPU diagnostic as **NO_TRAIN**. All five original states
reproduce their retained MAE within 1.26e-8 eV. The three retained prediction
bundles reproduce within 3.82e-6 eV maximum absolute difference. Final states
have trace-MAE guards only. All source, artifact, row, finite-value, unchanged
parameter and restored-buffer checks passed. No optimizer, gradients or training.

Each state uses the same 16,384 TRAIN members and 50,000 previously selected
development members. Calibration uses TRAIN features with dropout disabled,
18 BN modules and 128 batches. Units below are eV; lower is better.

| Retained LIVE state | Raw TRAIN | Raw development | Clean TRAIN | Clean development |
|---|---:|---:|---:|---:|
| Initial parent pass 49 | .061374754 | .104904075 | .056516516 | .103658948 |
| WD .001, selected tail pass 15 | .060393229 | .105829586 | .055473890 | .104811300 |
| WD .001, final tail pass 20 | .060093733 | .106447274 | .054299372 | .105134167 |
| WD .01, selected tail pass 15 | .060106354 | .105617390 | .055356491 | .104776548 |
| WD .01, final tail pass 20 | .060300821 | .106467514 | .054823561 | .105313709 |

## Fixed contrasts

Positive development error difference is worse. Paired-row 95% intervals use
the frozen 1,000 resamples/seed 42; they do not measure training stochasticity.

| Clean contrast against initial | TRAIN difference, meV | Development difference, meV [95% interval] |
|---|---:|---:|
| WD .001 selected | -1.043 | +1.152 [.900, 1.422] |
| WD .001 final | -2.217 | +1.475 [1.230, 1.703] |
| WD .01 selected | -1.160 | +1.118 [.854, 1.371] |
| WD .01 final | -1.693 | +1.655 [1.415, 1.891] |

Selected-to-final clean development worsens by .323 [.167, .482] meV for WD
.001 and .537 [.369, .696] meV for WD .01; sampled TRAIN fit improves in both.
All four tail endpoints remain worse after matching the BN procedure.

WD .01's selected advantage over WD .001 shrinks from .212 meV with original
BN to **.035 meV after Clean-BN**, interval for candidate-minus-reference
[-.216, .149] meV. Its clean final endpoint is .180 [.016, .336] meV worse.
There is no retained clean-state advantage that supports another stronger-WD tail.

## Disposition and costs

The exact buffer-only explanation fails. The retained tail recipe improves
in-sample fit while harming consumed-development performance, even with common
calibration. Preserve the original tail training decision as negative; no model
promotion, new selected endpoint, optimizer restart or further exposure is released.
The training run's independent strict-target evidence blocker is unchanged by
this diagnostic. This is not a training replay-ready claim.

Worker wall 647.321 seconds; measured process CPU 2,480.016 seconds. Outer wall
648.537 seconds includes process startup/teardown. Native wall and CPU are
separate units; no GPU/queue allocation. Preparation, tests, Git and closure are
outside these timers. Exact timings, predicates, predictions and completion
hashes are in [result](results/result.json) and [completion](results/completion.json).
