# K1 execution audit, 2026-10-11

Engineering-only CPU audit on the owning experiment branch. It does not change
the frozen attempt003 source, authorize another run, accept science, qualify
GPU numerical equivalence, or promote an execution default. No molecular role,
training dataset, remote runtime or accelerator was consumed by this audit.

## Reusable Changes

| Owner | Change and preserved boundary |
|---|---|
| `training_reproducibility.py` | Collect detached scalar metrics and transfer once per epoch, preserving the original sequential Python addition. Group finite-state flags by accelerator device without skipping any tensor check. Clean up failed atomic Torch-save temporaries. |
| `k1_screen_training.py` | Batch little-endian sampler hash encoding, with unchanged order/bytes. Transfer only model/loss inputs; keep row IDs, graph boundaries and development labels on CPU. Reject invalid development metrics before overwriting acknowledged state. |
| Same trainer, resume | Safe tensor-only loading in qualification and formal resume, using the existing RNG translator; bind last checkpoint bytes to the acknowledged trace. Reject missing or malformed epoch timing before new optimizer steps; do not fabricate historical cost. Validate retained selection before resume. |
| `experiment_family_workflow.py` | Share prediction/selection checks between terminal inspection and pre-resume validation. Pre-resume checking is not terminal acceptance or replay readiness. |
| `experiment_execution.py` | Pass the pinned target-identity binding to K1 completion directly, removing a duplicate terminal inspection. GPTrans retains its existing dispatch fallback. |

No model dimensions, layers, loss, optimizer selection, precision, scheduler,
clip threshold, row order, batch size or worker lifecycle changed in this audit.
The previously proposed fused optimizer/layout candidate remains separately
frozen by the experiment protocol; optimizer rounding is not strictly equivalent.
These changes live in shared `src/molgap/` owners, not a parallel experiment loop.

## Verification

CPU fixtures cover actual 40-epoch loop persistence, interruption/resume,
checkpoint hashes, selected artifact failures, timing corruption, invalid
development metrics, completion/dispatch binding, atomic-save failures and
deferred metric reduction. Synthetic K1 forward/backward/optimizer-step tests
cover layout weight/gradient/buffer/RNG parity. Real PyG metadata placement is
checked using a meta device; synthetic device flags exercise grouping logic.
Neither is a CUDA arithmetic or performance test.

Final related regression: 303 passed in 294.83s, covering reproducibility,
K1 loop/layout/transfer, dispatch, family output hooks, workflow/release inputs,
Kaggle bootstrap/push helpers and V4 runtime. A subsequent IO/RNG suite passed
28 tests in 0.68s, including the newly added safe RNG next-draw check; these
counts overlap and must not be added. One upstream PyG deprecation warning.
`compileall -q src/molgap` and `git diff --check` passed (Git only warned about
LF-to-CRLF normalization). Actual GPU preflight was not executed in this audit.

Sampler-only [measured CPU record](cpu_sampler_benchmark.json): three alternating
repeats, seed42, 40 epochs, 99,968 presented rows per epoch, exact matching hash.
Median scalar encoding 1.209247s versus batched encoding 0.922443s: 23.7% less
time for this startup function only. It is not a training speed improvement.

## Remaining Boundaries

- Native T4/A100 throughput and numerical behavior of the new execution changes
  are not measured. The live attempt003 measures only its older frozen package.
- Epoch-loader reuse is an existing separate helper, not enabled here; its RNG
  and PyTorch-version requirements need a matching frozen contract.
- Padded K/V projection work, projection fusion and alternate attention kernels
  may offer additional savings but can change GEMM/reduction arithmetic. They
  are not silently adopted as equivalent replacements.
- Development prediction batches still transfer to CPU individually. Deferring
  those copies requires a native memory-budget and throughput check.
- Checkpoint, trace, timing and selected artifacts are individually atomic, not
  one transactional bundle. Interrupted disagreement fails closed and needs
  reconciliation; it is not repaired by inventing observations.
- Pre-resume selection checks detect missing files, invalid values, identity
  mismatches and trace/metric disagreement. Before a terminal hash manifest,
  they cannot prove arbitrary finite selected-weight byte integrity without an
  independently retained selection digest. Terminal artifact SHA checks remain.
- Resume allocation totals remain unknown when earlier allocation ledgers are
  absent; timing observations do not justify fabricating device allocation cost.

This audit removes identified overhead and failure paths; it does not establish
that the execution is globally optimal or that all platform failure modes are
covered. Review and owner routing are required before desktop integration.
