# Execution40 delivery qualification — 2026-10-11

Kernel138147173/version3 completed both authorized forty-pass arms. Independent
[saved-output acceptance](gpu/results/acceptance.json) passed, including source,
initialization, fixed data, runtime certificates, FP32/noTF32, BS128, all40 trace
observations,31,240 updates and3,998,720 sample presentations per arm. All112
manifest-selected files were retrieved at the exact version with SHA checks;
[metadata](gpu/retrieval_metadata_v3.json) and
[artifact retrieval](gpu/retrieval_artifacts_v3.json) retain transport identity.
No model inference, local model training or protected-role access occurred
during acceptance. This was a delivery experiment / PAIRED_ENDPOINT, not a new
architecture claim or a sixty-pass completion.

## Measured result

| Quantity | Original | Optimized |
|---|---:|---:|
| Parameters | 5,871,201 | 5,871,201 |
| Complete training stage, seconds | 9,587.773 | 8,161.518 |
| Complete-stage presentations/s | 417.065 | 489.948 |
| First epoch, seconds | 271.081 | 220.034 |
| Median of epochs1..39, seconds | 248.530 | 202.546 |
| Selected EMA MAE, eV | 0.1426747491 | 0.1419957587 |
| Selected epoch, zero-based | 39 | 39 |
| Final live MAE, eV | 0.1478768736 | 0.1475237757 |
| Best live MAE, eV | 0.1473212242 | 0.1475237757 |
| Synchronized live+EMA evaluation, seconds | 2,211.056 | 2,491.533 |
| Resume-checkpoint publication, seconds | 11.047 | 11.029 |

Measured complete-stage throughput ratio was1.174754x (17.48% higher throughput;
14.88% lower stage wall). Candidate-minus-control selected error was
-0.0006789903eV; the paired-row95% interval was
[-0.0015226477,+0.0001732313]eV. The prospectively declared1.10x speed,
<=0.0005eV selected regression and<=0.001eV interval-upper-bound tolerances
passed. Both selected EMA endpoints were also final endpoints, so best-epoch
selection did not conceal a worse final EMA result. The best-live score did
not improve; the final-live score did. The interval includes zero and does not
measure seed variation: no supported architecture or robust accuracy gain was
established.

## Interpretation and limits

The opt-in backend qualified for this particular forty-pass local-bond GPTrans
delivery scope. It did not qualify500K/full execution, a different model family,
or substitution into a historical frozen reference contract. Preserve its
distinct backend/optimizer fingerprints and runtime certificate.

Both workers shared host resources. After optimized exit, original epochs33..39
fell to approximately182–185seconds, versus roughly241–270seconds during much
of concurrent execution. This is evidence consistent with shared-resource
contention, not an isolated attribution to a particular loader/CPU mechanism.
The measured1.175x ratio is a paired-allocation observation, not a proven
standalone hardware speedup. No speedup extrapolation to A100/DCU/full was made.
Evaluation consumed30.53% of optimized stage wall and was slower in aggregate
than original; the buffered evaluation change was not independently shown to
accelerate evaluation. CPU step-call timers encompass queued forward/backward,
optimizer and EMA calls, not synchronized GPU compute or AdamW alone. Resume
save timers do not cover every output publication.

Predictions were not numerically identical: mean absolute prediction difference
was0.07366495eV, maximum3.01283050eV. Small aggregate MAE drift must not be
confused with per-molecule equivalence. Foreach/layout changes were bundled;
the experiment cannot identify which caused trajectory drift or savings.

[Retention audit](gpu/results/retention_audit.json) verified all ten-pass chunks,
finite FP32 model/EMA state, optimizer step counters, scheduler cursor, RNG
schema, source/certificate identity, last-checkpoint identity and selected
model equality to final EMA. Checkpoints were inspected as data, never resumed
or executed. Optimizer-preflight allocated peaks were approximately1.00GiB for
both arms; training-stage peak memory was unmeasured, not inferred from preflight.

## Cost and evidence qualification

The version3 Python allocation window was9,657.550seconds across two T4s,
5.365305712 allocated T4-hours, including the idle-device tail. Retained v1/v2
failures add0.007408938 and0.044874247 T4-hours: campaign total5.417588896 native
allocated T4-hours. Provisioning outside Python and exact platform billing were
unmeasured. Costs were not substituted with per-arm training-stage sums.

The [compact native snapshot](gpu/results/evidence_snapshot.json) preserves raw
trace/manifest/calibration content and source-file hashes. Model/checkpoint and
prediction payloads remain in local platform records, not Git.

Canonical RML finalization remained **blocked**, not complete/replay-ready.
The [explicit closure gap](gpu/rml_closure_status.json) records empty prospective
reference IDs and the absent same-run-pair binding. Existing trace finalization
cannot accept this noncausal same-run control without an appropriate reviewed
retention path. Original plans were not rewritten, historical reference claims
were not fabricated, available training traces were not hidden, and shared
strict replay gates were not weakened. No retraining was justified by this
metadata limitation. Native evidence and the delivery result remain valid
within their stated scope; canonical RML closure requires separate repair.

The acceptance-only repair allowed at most2ULP for Windows recomputation of
Linux cosine LR values, while requiring raw/canonical recorded LR equality
and exact counters. At epoch18 the cross-platform discrepancy was one last bit;
no recipe or remote source was changed. Focused backend tests passed35/35.

The monitor event was closed without a successor. The existing Luna heartbeat
was paused. No extra seed, training extension,500K/full run or new experiment
was submitted.
