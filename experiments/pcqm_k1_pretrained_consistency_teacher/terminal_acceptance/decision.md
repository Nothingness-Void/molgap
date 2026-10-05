# K1 pretrained consistency / mean teacher decision - 2026-10-05

Both arms completed40epochs,31240updates and3998720sample presentations and
passed the existing mechanical inspector with the frozen float32 target binding.
Actual Kaggle1 kernel137071131/version1 is COMPLETE; both assigned T4 workers
passed the all-arm qualification barrier before training. No continuation.

| Arm | Clean live development MAE(eV) | Selected epoch |
|---|---:|---:|
| A pretrained consistency |0.138265848|40|
| B pretrained consistency +mean teacher |0.136781212|40|
| Fixed retained equal teacher |0.134665993|Retained constituent selections|

B improves1.484636meV over same-job A. The1000draw seed42 paired-row bootstrap
gain interval is[0.759455,2.196867]meV. The predeclared1meV point/positive-bound
teacher-increment gate passes. A is a completed control(CLOSED); B is
POSITIVE_UNDER_CONTRACT for this narrowly defined teacher-increment question.

B loses2.115219meV to the retained fixed teacher, exceeding the separate1meV
compression-loss allowance. Compression fails. Do not report B as an equivalent
single-model replacement, an adopted best model, or qualified500K/full handoff.
No scale-up, sweep or new training is released by this acceptance.

The same-role saved-prediction comparison is exact on50000development rows.
These rows have selection history; row bootstrap is not training stochasticity.
Both arms share the same reused10-pass backbone and reset head; their contrast
isolates the declared teacher objective, not the benefit of pretraining.

Training invocation allocated-T4 lower bounds are8966.101s(A) and8942.644s(B),
4.974651T4hours combined. Diagnostic windows are retained separately. These
exclude bootstrap/queue and reused historical pretraining/teacher computation;
they are not GPU busy time or complete lineage cost. CPU/queue remain unknown.

Existing V5 comparison classification and RML finalization own strict/replay
status; read[closure receipt](closure_receipt.json) after publication. Evidence
remains on the owning experiment branch pending a reviewed adoption/Git route.
Read[attribution](attribution.md) and[exact metrics](scientific_metrics.json).
