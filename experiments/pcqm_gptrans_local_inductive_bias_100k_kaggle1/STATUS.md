# GPTrans local inductive bias 100K status

**Submitted to Kaggle1; initial authoritative status: QUEUED.** The existing
Kaggle adapter accepted kernel
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v1`, with no
invalid source references. The first subsequent Kaggle status query returned
`KernelWorkerStatus.QUEUED`. Exact source, package, Spec, graph, and arm
identities are frozen in [release_gate.json](release_gate.json); the local
reconciled launch receipt and observed queue state are in [launch/](launch/).
No runtime certificate, training result, or terminal RML evidence is yet claimed.

The planned Kaggle1 T4x2 job contains two new arms: `rwse16` (A) and
`rwse16_local_edge` (B). A is B's same-job relative reference; the
accepted historical GPTrans-T 100K evidence is contextual only. The intended
gate and acceptance requirements are in [protocol.md](protocol.md).

Both arms passed the real-shard CPU model-smoke check before launch. The Kaggle
T4 workers must still independently pass their GPU runtime certificates before
training. Reconcile the same kernel after terminal state; retrieve only the
artifacts needed for each arm's acceptance, then close both RML trajectories.
Neither arm is replay-ready from submission or queue state alone.
