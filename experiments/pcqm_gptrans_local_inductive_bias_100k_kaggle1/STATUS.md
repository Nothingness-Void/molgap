# GPTrans local inductive bias 100K status

**Contract frozen; not submitted.** No Kaggle kernel, source package, frozen
ExperimentSpec, launch receipt, result, or terminal RML evidence is claimed by
this record. The source commit, package, per-arm model-state, Spec, and run
identities must be published in a separate release gate after verification;
[training_contract.json](training_contract.json) owns the numeric recipe only.

The planned Kaggle1 T4x2 job contains two new arms: `rwse16` (A) and
`rwse16_local_edge` (B). A is B's same-job relative reference; the
accepted historical GPTrans-T 100K evidence is contextual only. The intended
gate and acceptance requirements are in [protocol.md](protocol.md).

Before release, finish the exact model/initialization freeze, prospective RML
plans and same-run binding, real-shard preflight, and independent T4 runtime
qualification. Record the authoritative remote identity and state here only
after an actual submission. Record each arm's terminal evidence separately;
do not label either replay-ready from a queue or completion status alone.
