# Formal-reference CPU release checks (2026-10-18)

The human authorized the two unchanged model lineages, accepted100K/Common-Dev,
seed42, physical BS128,60 complete passes and an aggregate9 T4-hour ceiling.
The [execution guide](../../../docs/operations/MODERN2D_FORMAL_TRAINING.md)
defines the new admission boundary and the context-only interpretation.

Configured Luna workers, not the coordinating agent, executed the checks:

| Scope | Final observed result | Retained evidence below `E:/MolGap/artifacts/modern2d-formal-screen/` |
|---|---|---|
| Qualification terminal adapter |16 passed; separate finalize fixture1 passed |`luna-rml/verification-20261018/` |
| Formal Recipe/Session/admission |58 passed |`luna-training/formal-fixed-32f08fd003f64cf0a4a365f547774686.xml` |
| Declaration builder |14 passed; actual accepted metadata/proof read-only validation passed |`luna-prepare/test_modern2d_training_prepare.junit.xml` |
| Budget/runtime/release deltas |17 passed |`luna-release/formal-release-rerun.xml` |
| Existing prospective/comparison/workflow |93 passed,1 skipped |`luna-release/workflow-focused.xml` |
| Existing runtime/bootstrap/mount/registration |73 passed,2 skipped |`luna-release/runtime-bootstrap-focused.xml` |

The suites overlap other retained regression runs; their counts must not be
added into a claimed unique full-suite total. The larger training-worker run
observed257 passed and2 failures before the fixture repair; it was not an
all-green run. Both failing formal cases subsequently passed in the58-case
focused rerun. AdamW reused input checkpoint moment tensors, so continuing the
test mutated the in-memory evidence's optimizer steps while its saved cursor
stayed at1. The fixture now loads a deep copy. The production resume validator
was not relaxed. Initial failure logs remain retained.

The tests proved additive preview blocking, fresh formal certificate binding,
actual tiny-model optimizer/scheduler/RNG/cursor repeat/resume, read-only Dev,
role/source/config tamper rejection, all-arm preflight barrier, two assigned
devices, pinned Torch2.7.1 bootstrap and graceful cost pause. They did not prove
native60-pass throughput, a complete9-hour billed cost, MAE improvement or
desktop migration. Host skips are not accepted capabilities.

The Common-Dev identity worker read only the47 accepted Dev chunks and source
members, using approximately301MB sampled peak RSS. The50000-row target raw
float64 SHA was `0971904c41fe9f2abaa06f895bd4a7120b1fc92335010911862118bdb6ec4701`.
No graph/cache was rebuilt, no training role/model/CUDA was instantiated, and
no protected role or label statistics were opened. The original independently
hashed [proof](../development_target_proof.json) and compact
[target manifest](../target_manifest.json) were retained separately.

All14 pre-existing dirty user byte pins matched after the focused checks.
The prospective publisher now defers derived-cache rebuilding when those views
are dirty; canonical prospective publication and release checks remain required.
CPU qualification is `INFRA_CPU_ACCEPTED`, not remote scientific acceptance.
