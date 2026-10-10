# Packed loader verification

2026-10-10. Read the [accepted decision](loader_packed_attempt_001/terminal_decision.md)
for actual TRAIN input, source identity, measured windows and limits. Its finalized
evidence is immutable; this note records subsequent infrastructure regression.

The project venv with this worktree's `PYTHONPATH` and CUDA visibility `-1` ran
ten focused test files:396 passed,3 skipped,7 warnings in156.60s. Coverage includes
the original screen trainer, shared packed reader, loader reuse/profiler, source
package/workflow, CPU preflight, saved closure and sampler/JSON sender proposals.
Warnings are upstream PyG deprecation and expected unavailable CPU pinning.
Skipped tests are not GPU qualification. The earlier detached terminal handle
was unavailable; this count belongs to the observed repeat, not that lost output.

`research_memory validate` passed; `check --frozen` confirmed generated outputs
are current after the packed diagnostic's finalization/rebuild. Whole-library
portable acceptance is not claimed: historical locally retained ignored inputs
are still a separate custody limitation.

The reader is module-scope for Windows spawn/pickle and retains the existing
two-item PyG payload/separation behavior. Both selected K1 reader callers reuse
it; screen forbidden-field stripping remains in its original owner. Source
inventory and K1 serialization dependencies include the reader, and the package
path exception admits exactly this reviewed file, not arbitrary `dataset` paths.

The accepted packed diagnostic uses4096 TRAIN graphs only, not500K loading or
model execution. Native Linux timing, GPU consumer/pinning, checkpoint-step
resume, retrained quality and adoption remain unqualified. Registered100K's
global-RNG compatibility is a separate infrastructure extension; it cannot
retroactively enlarge this frozen diagnostic's qualification.
