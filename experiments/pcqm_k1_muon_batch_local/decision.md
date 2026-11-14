# Local diagnostic decision

2026-11-06. Retain the batched FP32 Muon execution candidate; do not change
the registered optimizer or any running/frozen remote source package.

The116hidden matrices form10equal-shape groups. Instead of1,740separate matrix
operations, the candidate submits150batched matrix operations per optimizer
update, with the same individual matrix work and five Newton-Schulz iterations.
It retains per-parameter momentum, step, Nesterov, scaling, decay and AdamW
ownership; no reduced precision or new model/recipe parameter was introduced.

## Measured results

RTX5060, Torch2.7.1+cu128, FP32/noTF32. Each entry averages the medians of two
ABBA windows with four warmup and24measured updates per window, profiler off.

| Scope | Reference ms/update | Batched ms/update | Time reduction | Throughput ratio |
|---|---:|---:|---:|---:|
| Optimizer, actual K1 shapes and synthetic gradients |65.521|9.321|85.77%|7.03x|
| Existing FamilyKernel,128synthetic graphs/batch |119.959|64.729|46.04%|1.85x|

Shared-kernel timing includes layout/transfer, forward/backward, clip, optimizer,
EMA and schedule; excludes checkpoints, development evaluation and input loading.
This is not a measured100K/500K epoch or T4/A100 speedup. The GPU process's
measured diagnostic interval was24.776seconds. That interval excludes interpreter
startup and the separate runtime download/setup; device busy time was not metered.

## Numerical checks

The three affected CPU tests passed, including the original update formula,
unchanged native AdamW subgroup, zero/None gradients and old/new state interchange.
CPU multistep maximum parameter/momentum errors were1.19e-7/2.98e-8.

CUDA checks use identical initial weights, gradients, RNG and input sequence:

| Check | Maximum absolute difference |
|---|---:|
|20fixed-gradient optimizer updates, model and optimizer state |2.831e-7|
|28shared-kernel scratch updates, loss sequence |5.960e-8|
|Final synthetic predictions,128rows |1.937e-7|
|Final model state, including BN buffers |1.026e-6|

The implementations are numerically close, not bit-exact. No molecular MAE,
long-run quality, native T4 qualification or scientific promotion was measured.

## Retained evidence and repair

Raw results, initial state and exact executed source snapshot:
`D:/molgap-artifacts/local_profiles/k1-muon-batch-20261106-attempt002/`.
The compact result is also beside this decision. The prospective trajectory
was written before scratch execution. This diagnostic is not formal training
and does not grant a scientific replay-ready entry.

Attempt001 performed no optimizer updates: the shared Torch2.7.1+cu126 lacks
RTX5060 kernels. It remains under the first external output directory.
Attempt002 used the original project interpreter with an isolated cu128 package
on PYTHONPATH; the shared environment was not modified.

Review found that optimizer-only peak allocation included a post-timing state
snapshot. Those memory figures are excluded from interpretation. The sampler
now reads peak allocation before that snapshot; timings and numerical results
are unaffected and were not rerun for this unused memory field. The raw capture
remains unchanged, with its original source snapshot/hash bindings.

Next useful evidence is a separately authorized fixed molecular quality comparison
using this candidate. Do not overwrite the running500K Muon implementation or
add another trainer, submission path, configuration layer or optimizer grid.

## Post-hoc repair note

`run.py` hash differs from the recorded one: `result.json` and
`trajectory.json` record `e6f8860d...`, the current file is `e71448af...`.
The difference is the post-hoc sampler fix described above. The run was not
repeated. No molecular-quality comparison exists for this candidate, and it
must not be wired into any production default.
