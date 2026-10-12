# Execution tuning v3 delivery qualification — 2026-10-12

Kaggle2 kernel138254522/version1 completed both authorized forty-pass arms.
Independent [saved-output acceptance](gpu/results/acceptance.json) passed the
frozen [protocol](protocol.md). Exact-version [metadata](gpu/retrieval_metadata_v1.json)
and [artifact retrieval](gpu/retrieval_artifacts_v1.json) retained all118 selected
files with SHA checks. Acceptance executed no model, training or checkpoint resume,
and accessed no protected role.

## Result

| Quantity | Original | Optimized v3 |
|---|---:|---:|
| Parameters | 5,871,201 | 5,871,201 |
| Complete training stage, seconds | 8,366.348 | 7,381.791 |
| Complete-stage presentations/s | 477.953 | 541.701 |
| Selected EMA MAE, eV | 0.1426747491 | 0.1419957587 |
| Selected epoch, zero-based | 39 | 39 |
| Median epochs1..39, seconds | 212.717 | 182.474 |
| Final live MAE, eV | 0.1478768736 | 0.1475237757 |
| Live+EMA evaluation, seconds | 1,900.838 | 1,561.752 |
| Resume-checkpoint publication, seconds | 10.300 | 11.034 |
| Training peak CUDA allocated memory, MiB | 1,029.218 | 1,051.324 |

Complete-stage throughput increased13.34% (1.133377x); stage wall decreased11.77%,
approximately16.41minutes. The frozen >=1.10x speed criterion passed. Selected
EMA candidate-minus-original error was -0.0006789903eV, with paired-row95%
interval [-0.0015226477,+0.0001732313]eV. Both accuracy-tolerance gates passed.
The interval includes zero and does not measure seed variation; this was an
execution delivery qualification, not a supported architecture accuracy win.
Both selected EMA endpoints were also final endpoints.

## What the counters establish

The optimized arm used one Torch thread and one training worker, versus original
two threads/four workers. Both shared the same observed4-CPU host quota. Its
256MiB development cache retained159,776,160bytes: one cold pass,79 warm passes,
no overflow. EMA built one tensor plan for31,240 updates; live-weight backup
allocated once (23,484,804bytes). These establish reuse, not individual causal
speed contributions: the execution changes were bundled.

Evaluation took17.84% less time and21.16% of optimized stage wall. Checkpoint
publication was approximately0.15% of that stage: weakening durability was not
justified. The optimized arm's allocated-memory peak increased about22.1MiB;
there was no demonstrated training-memory reduction. Process-local dataset
caching reported two loads and zero hits because preflight and training were
separate processes; no deserialization saving was established for that feature.

The [retention audit](gpu/results/retention_audit.json) verified eight ten-pass
chunks, finite FP32 model/EMA tensors, optimizer counters, scheduler cursors,
RNG schema, source/runtime/scientific identities, last-checkpoint hashes and
best-model equality to final EMA. Each arm performed31,240updates and3,998,720
sample presentations, with unchanged40-of60 schedule scope.

All50,000 optimized-v3 selected prediction values, row IDs and targets were
bitwise identical to the preceding optimized-v1 run's retained predictions.
Thus the added cache/transfer/EMA-reuse layer introduced no observed selected
prediction drift relative to optimized-v1. This does not assert bitwise equality
to original: the earlier bundled optimizer/layout changes already produced a
different training trajectory. Cross-job timing cannot isolate v3's incremental
speedup; the preceding job was not a concurrent optimized-v1 control.

## Scope, cost and closure

Both arms ran concurrently on T4x2. Original epochs35..39 fell to approximately
173–175seconds after optimized exit, versus approximately210–216seconds during
much of concurrent execution. This is consistent with shared-resource contention;
the1.133x ratio qualifies this allocation, not standalone GPU speed, A100/DCU,
500K/full, other model families or substitution into frozen scientific references.
CPU step-call timing includes queued forward/backward/optimizer/EMA work and is
not synchronized GPU compute or AdamW-only timing.

The Python allocation window lasted8,421.937seconds (2h20m21.94s), across two
T4s:4.678854 allocated T4-hours, including install/preflight and the idle-device
tail. Provisioning outside Python and exact platform billing were unmeasured.
The [native snapshot](gpu/results/evidence_snapshot.json) retains compact JSON
evidence and hashes; model/checkpoint/prediction payloads remain in platform
record storage rather than Git.

The opt-in v3 layer qualified only for this delivery scope. Both prospective
plans froze their same-run binding before compute, unlike the preceding assay.
Nevertheless canonical RML finalization/replay remained blocked by the common
finalizer's strict-comparison requirement for same-run training pairs. This
noncausal delivery experiment may claim PAIRED_ENDPOINT, never STRICT_CAUSAL.
The [closure status](gpu/rml_closure_status.json) preserves that limitation;
plans, strict gates and scientific claims were not rewritten to force admission.
Complete retained evidence is not a training Replay-Ready claim.

The controller committed the terminal interpretation and closed the monitor
event/binding. The existing Luna heartbeat was paused. No successor, extension,
extra seed, scale-up or new training was submitted.

## Local verification

Eight focused tuning-release/retrieval tests passed. Repository RML validation
and `check --frozen` passed in the working checkout. Unrelated pre-existing
working-tree RML/derived edits were preserved and were not included in this
experiment's commit; this check does not certify a different clean checkout.
No local model training or inference was executed.
