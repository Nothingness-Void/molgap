# Shared execution refinement

Date: 2026-10-15. Scope: execution maintenance on the existing family core.
Base: desktop `853ac0ed`. No model, objective, precision, batch, sampler,
schedule, EMA decay, development cadence or checkpoint cadence is changed.
This is not a new architecture result, resource release or measured speed claim.

## Shared owners

| Change | Owner and invariant |
|---|---|
| Scoped GPTrans training safety-fill suppression | `family_training_kernel.py` reuses `deterministic_memory_fill_context`; deterministic algorithms remain enabled, and the previous fill setting is restored on all exits. Evaluation and checkpoint validation keep their normal policy. |
| Native fill-on reference | `family_training_runner.py` adds one two-update training-only scratch reference. Its `fill-enabled.pt` envelope binds source/run context, recipe, runtime, reference role and actual before/after-update fill/determinism observations. The inner model/optimizer/EMA/schedule/RNG fingerprint must match the accelerated repeat. Renaming a repeat artifact, missing or changed evidence blocks training; there is no slow fallback. |
| In-place local-bond accumulation | `gptrans_bond_local.py` accumulates into its own newly zeroed buffer, avoiding the functional index-add output copy. Directed edges, reduction order policy and trainable tensors are unchanged. |
| One complete producer validation | Registered sessions pass a synchronous snapshot builder to `SessionOutputs.checkpoint_from_builder`. `write_resume_state` validates the complete payload once, including optimizer, EMA, scheduler, cursor and RNG tensors. Public `session.state_dict()` and generic callback sessions retain their independent checks. No caller-supplied validation boolean or cached authority is introduced. |
| One evaluation weight view | `FamilyKernel.evaluate(..., on_evaluation=...)` permits synchronous selection while the selected live/EMA view is installed. It restores live state even if evaluation/selection fails. The observer is trusted Python wiring, never a JSON-selected callback. |
| Profiling continuity | `family_execution_profile.py` observes the builder and publisher and adds `finite_validation`; the existing synthetic diagnostic uses the registered sink when the selected frozen source supports it. |

New registered source packages inherit these changes automatically. Historical
payloads and running jobs keep their frozen code. New source/package and native
qualification remain mandatory, even when algorithm recipe bytes are unchanged.
The additional reference consumes scratch work, not formal training exposure;
its cost belongs to the existing native preflight interval. Retain its PT file
alongside the existing repeat/resume artifacts and certificate.

## Verification boundary

`tests/test_gptrans_execution_refine.py` covers synthetic CPU values, gradients,
optimizer/EMA/RNG equality, empty/repeated-target bonds, complete-state rejection,
checkpoint validation count, EMA-view reuse and retained-reference tampering.
The existing family/session, recovery, source packaging, logging and profiler
suites remain the regression owners. CPU success is not native GPU qualification.
The same tiny-graph equality cases optionally run on CUDA with
`MOLGAP_EXECUTION_TEST_DEVICE=cuda`; this requires separately available/authorized
local hardware and does not use real molecules or qualify a remote platform.

Dataset-level throughput and a new complete molecular old/new comparison have
not been measured. Do not reuse the earlier T4 15.7%
observation as the measured effect of this source. The prior accepted comparison
remains at [its owner](../../experiments/family_topology_t4_100k/terminal_acceptance/decision.md).

Checkpoint finite checks now reside in the producer's publication interval.
The raw `checkpoint_state` phase no longer includes those checks on that route;
compare whole transactions or non-overlapping host-self phases, not a single
renamed timing component. The new `finite_validation` range makes this visible.

## Deliberately not changed

- SPD/degree per-molecule cache: needs bounded storage, fixed-row/cache identity
  and CPU-versus-device measurements; blindly moving it to the CPU may regress.
- QKV/1x1-convolution layouts and fusion: the retained top-50 profile has no
  shapes, so extra copies and numeric consequences need a bounded native probe.
- FlashAttention, AMP/TF32, fused optimizer, bucketing and fewer evaluations:
  not equivalent unqualified replacements for this frozen training recipe.

For another authorized timing diagnostic, reuse
[the shared profiler](FAMILY_EXECUTION_PROFILE.md) or
`family_execution_diagnostic.py`, publish its prospective scope first, and
separate compute, checkpoint, development and allocated-device costs.

## Local acceptance

Independent executors: `gpt-6-luna`, `max`; project desktop venv. All final
results below supersede intermediate failed fixture/wrapper runs. Source edits
were frozen for the final checks; subsequent edits record this receipt only.

| Final group | Result | Seconds |
|---|---|---|
| `test_gptrans_execution_refine.py`, CPU-only | 40 passed | 5.80 |
| Kernel/session/runner/profiler, hotpath/tuning, V4, logging, diagnostic/recovery | 251 passed | 21.34 |
| Full lifecycle, core, recipe, workflow, data and loaders | 186 passed | 206.13 |
| Explicit CUDA selector, RTX 5060, `-k cuda` | 8 passed, 40 deselected | 22.28 |

These are 477 CPU plus 8 distinct CUDA cases, zero failures/skips. CPU jobs
masked CUDA/HIP/ROCR with `-1` and asserted zero visible devices. The separate
CUDA invocation observed exactly one RTX 5060, 1% initial utilization and over
6 GiB free; its process exited normally in 33.47 seconds, below its 180-second
ceiling. Only synthetic graphs and fresh random tensors were used. Six GPTrans
modes match fill-on/off values, gradients, optimizer, EMA and RNG exactly;
two scatter cases cover empty and repeated targets. Local CUDA success is not
a reusable T4/A100 certificate or a full-dataset MAE claim.

Changed-module `compileall` and `git diff --check` passed. Independent
`gpt-6.1-sol`, `medium` review found the copied-repeat admission gap in the
initial patch; the producer-bound envelope and corruption tests repair it.
Final review reported no actionable findings. No real molecular or protected
role data was consumed, no remote job was launched, and no canonical research
record was modified.
