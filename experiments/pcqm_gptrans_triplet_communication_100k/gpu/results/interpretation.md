# Triplet communication: accepted 100K result — 2026-10-09

## Decision

Both arms passed the prospectively frozen single-seed nomination gate against
the qualified uncapped local-bond parent. Retain both implementations and
evidence; prioritize aggregation for a separately authorized transfer study.
Query-conditioned attention did not establish an advantage over aggregation.
Neither result established seed stability, 500K transfer, full-scale ranking or
delivery superiority. No successor, extra seed or protected evaluation was
released by this terminal decision.

[Protocol](../../protocol.md) owns the mechanisms and scientific contract.
[Independent acceptance](acceptance.json) owns the recomputed endpoints and
paired analyses; [saved-output analysis](analysis.json) owns curve/diagnostic
summaries and the direct attention-versus-aggregation comparison.

## Matched observations

| Arm | Parameters | Selected EMA MAE (eV) | Gain against parent (eV) | Selected epoch, zero-based | Mean recorded epoch (s) |
|---|---:|---:|---:|---:|---:|
| Frozen local-bond parent | 5,871,201 | 0.1423582275 | — | Reference-owned | Reference-owned |
| Third-pair-gated aggregation | 5,880,961 | 0.1393084501 | 0.0030497774 | 38 | 290.0442 |
| Query-conditioned triplet attention | 5,889,409 | 0.1393149375 | 0.0030432900 | 46 | 307.2497 |

The paired-row 95% intervals for candidate-minus-parent were
[-0.0039477750, -0.0021774815] and [-0.0039169554, -0.0021277864] eV.
Both point estimates exceeded the protocol's 0.003 eV material gate and both
intervals favored the candidate. Neither interval established a gain of at
least 0.003 eV, and row resampling does not measure training-seed variability.
The small margins above the policy threshold should not be overstated.

Attention-minus-aggregation was +0.0000064874 eV, with interval
[-0.0008484977, +0.0009294359]. Accuracy superiority was not established.
Its observed mean epoch time was 1.05932 times aggregation's in this notebook;
this is an execution observation, not isolated algorithmic profiling. The
aggregation arm added 9,760 parameters; attention added 18,208. Parameter growth
did not measure the dense contractions' runtime or asymptotic memory costs.

## What the trajectory supports

Aggregation beat the parent's EMA metric at 59/60 aligned exposure points;
attention did so at 57/60. Both completed exactly 46,860 optimizer updates and
5,998,080 presentations. All four inserted returns had positive preflight
gradients and finite, nonzero terminal gradients/updates, including the final
virtual-pair route. This was not the earlier disconnected pair-transition
failure. No trained baseline or additional target information was introduced.

Both arms' last-ten-pass EMA changes were unfavorable (+0.0001997799 and
+0.0002810061 eV), while final training MAE reached about 0.0713 eV. Their
selected development endpoints preceded the final pass. This argues against
an automatic extension merely to obtain more exposure; it does not prove the
mechanism's convergence behavior at larger data scales.

Post-hoc baseline-error quintiles showed trade-offs between small and large
reference residuals, not uniform improvement. Those bins use ground truth and
condition on the reference's error; regression-to-the-mean effects are possible.
They are not chemical specialist identities, deployable routing features or
causal explanations of which molecules benefited.

The bounded result supports testing direct relation-to-relation communication
on this qualified parent. It does not show that query matching is necessary,
that TGT's complete recipe was reproduced, or that physical bond angles were
learned. Geometry, distance prediction and TGT pretraining remained absent.

## Execution, evidence and reuse

Kaggle2 kernel 137723341/version1 completed both isolated T4 workers. Recorded
notebook wall time was 18,586.4088 s (5.16289 h), with 10.32578 allocated
T4-device-hours, below the seven-hour ceiling. Optimizer preflight estimated
4.14130/4.31839 training hours and recorded more than 90% reserved-memory
headroom. Its utilization snapshots were not sustained utilization measurements.

The version-specific retriever retained only required JSON, predictions,
models and resume chunks. One transport error was resolved by idempotent
retrieval of missing files; no training job was resubmitted. Native acceptance
verified source/receipt, initialization, fixed data, runtime certificates,
selection, roles, counters, diagnostics and every required artifact hash.
Only saved prediction tensors were loaded locally; no model inference or
training occurred. Official validation, test-dev and challenge remained untouched.

Native per-arm terminal ingestion completed both prospective RML transactions.
[Actual Replay admission proof](replay_pair_proof.json) verifies two complete
STRICT_CAUSAL candidate/reference pairs against the unchanged qualified parent.
Both carry the frozen 100K POSITIVE_UNDER_CONTRACT label. This is screening
truth, not scale-promotion truth. Existing user edits and historical
reconciliation were preserved; unrelated global derived changes were not
included in this experiment's commit.

The existing Luna heartbeat was paused after its single terminal handoff. The
controller accepted the event and closed its custody binding. No new monitor
or conversation was created. A useful subsequent question would be whether the
unchanged aggregation benefit survives the later cohort and matched 500K
optimization; that requires separate authority and evidence, not an inference
from this screen.
