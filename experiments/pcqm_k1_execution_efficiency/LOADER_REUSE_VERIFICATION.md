# Loader reuse verification

2026-10-10; optional CPU implementation, no active sender or default adoption.
Read the [accepted real-data decision](loader_reuse_attempt_001/terminal_decision.md)
for timings and scope. Source commit `29b37538` was frozen before the diagnostic.

Parent regression:660 passed,3 skipped,7 warnings across20 focused files using
the project venv, this worktree's `PYTHONPATH` and CUDA visibility `-1`.
Skipped tests are retained in pytest's result, not claimed as GPU qualification.
Warnings are an upstream PyG deprecation and expected unavailable pinning on CPU.
No GPU model work, remote job, scientific training or protected role was used.

Five additional scalar synthetic-graph tests compare the original frozen-style
`loader(Subset(...))` path with original list-sampler loader reuse, using epoch,
processed cursor and dropped tail from `EpochPermutationBatchSampler`. Full
batch hashes agree, including an exhausted cursor. These are CPU infrastructure
tests, not molecular predictions or GPU checkpoint-resume certification.

Both shared CPU child entrypoints now request CUDA visibility `-1` before
imports; CPU/GPU mapping, parent environment and HIP/ROCR settings have targeted
regression coverage. Actual CUDA unavailable/count0/uninitialized observations
belong to the separately retained loader diagnostics, not their mock tests.

The observed-cost receipt from loader_attempt_003 used the earlier closure's
generic `attempt-001` fallback despite its correct unique trajectory/run/source
identity. Its immutable receipt is retained as generated, not silently rewritten.
Future loader closure now consumes the single canonical action attempt ID and
rejects missing/ambiguous IDs; the new reuse receipt retains `loader-reuse-attempt-001`.
Native cost and scientific classification are unchanged by that bookkeeping fix.
