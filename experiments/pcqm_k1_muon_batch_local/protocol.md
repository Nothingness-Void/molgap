# FP32 Muon batching: local execution diagnostic

2026-11-06. User-authorized local test, not molecular training or remote release.

Question: can equal-shaped K1 hidden matrices share batched Newton-Schulz
operations without changing the optimizer algorithm or per-parameter state?
The accepted Muon100K cost review identifies 116 matrices and 1,740 matrix
operations per update. RML lookup found that scientific experiment only, not
an existing batched execution comparison.

Reference: the existing `MuonAdamW` at desktop adb45e32. Candidate: a separate
unregistered batched subclass in the same owner. Keep FP32/noTF32, five
Newton-Schulz iterations, coefficients, per-matrix normalization, momentum,
Nesterov, parameter partition, weight decay and learning-rate scaling unchanged.
Keep per-parameter step and momentum state loadable by the reference. Production
defaults and frozen remote source packages stay unchanged.

Reuse the K1 constructor, shared FamilyKernel, ContractSchedule, synthetic graph
fixture, deterministic setup and atomic report/trajectory helpers. No molecular
cache, development labels, official roles, Kaggle, IMS or SCNet access.

One local RTX5060, at most 600 seconds elapsed within this diagnostic process.
Use synthetic graphs and seeded gradients on the actual K1 parameter shapes.
Run optimizer-only and shared-kernel update windows in A/B/B/A order; each window
warms four updates and measures 24 synchronized updates. Profile off during timing.
Use 20 fixed-gradient updates for numerical comparison and old/new optimizer
state interchange. Compare model, momentum, AdamW state, losses and synthetic
predictions; record exact equality separately from numerical error. Unit tests
cover zero/missing gradients and state interchange before this GPU exercise.

This is a throughput and numerical diagnostic, not a molecular-MAE claim.
Do not loosen a failed equality check or automatically adopt the candidate.
A faster candidate with small FP32 rounding differences needs a separately
authorized molecular quality comparison before live optimizer substitution.

Retain one source snapshot and raw result outside Git, plus a compact report.
Publish the prospective trajectory before any scratch optimizer/GPU execution.

Attempt001 failed before an optimizer update: the primary environment contains
Torch2.7.1+cu126, whose kernels do not support this RTX5060. Its source/report and
prospective record are retained under the first external output directory.
Attempt002 uses the same project interpreter with an isolated Torch2.7.1+cu128
package on PYTHONPATH; the shared environment and remote packages are unchanged.
