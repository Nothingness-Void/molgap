# Version2 bounded-stage acceptance - 2026-10-09

Exact kernel `nothingnessvoid/molgap-k1-consistency-500k-pair-s42-v1`,
ID137483676/version2, scheduler COMPLETE. Remote source bytes, T4 request,
mount membership, package/Spec and resume-source identity match retained release.
This is successful bounded execution, not the60-epoch scientific endpoint.

[Inspection](inspection_report.json), [retrieval](retrieval_manifest.json),
[scheduler](scheduler_snapshot.json), [source](remote_kernel.json) and
[native cost](evidence/invocation_cost.json) own observed values.

Both arms reached46/60,179676steps,22998528presentations. The original23-epoch
trace prefixes are unchanged. Required retained artifacts match producer SHA256;
runtime certificates, finite selected predictions and model states, selected
trace/checkpoint metrics, optimizer/RNG and epoch-boundary cursors pass inspection.
No per-epoch predictions were retrieved. Source rows are exactly500000:550000;
both arms have equal finite targets. Protected roles remain unconsumed here.

| Arm | Selected epoch, one-based | Selected MAE eV | Last epoch MAE eV |
|---|---:|---:|---:|
| coefficient0 mean2 |44|0.105171956|0.107178986|
| coefficient0.1 consistency |34|0.105595604|0.106862418|

The manifest's float32 metrics and CPU float64 recomputation agree within1e-7.
Independently selected prefix checkpoints favor coefficient0 by0.000423648eV;
the same last epoch favors consistency by0.000316568eV. Neither is the frozen
complete-endpoint comparison. Do not merge selected and last metric semantics.

V2 training allocation is17.452282796T4hours; cumulative V1+V2 training is
34.841851891T4hours, leaving17.158148109 under52. V2 bootstrap allocation is
17.562581289T4hours and includes its training window: do not add both totals.
Queue, outside-process allocation and CPU-core cost remain unknown.

Disposition ACTIVE_PARTIAL_STAGE. Fourteen epochs remain per arm, followed by
the predeclared identical clean-BN selected-state analysis and independent
terminal/RML closure. Neither arm is training replay-ready. This acceptance
did not train, infer, calibrate, publish a recovery dataset or submit another run.
Continuation requires a separately reviewed same-run release from epoch46,
not a restart from23. Owner branch remains open; no model adoption.
