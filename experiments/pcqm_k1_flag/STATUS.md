# K1 FLAG handoff

Owner: `codex/exp/k1-flag`; checkout mapping is in desktop BRANCHES.md.
User authority: detailed local RML review, one justified pure2D EdgeState-family
single-model experiment, Kaggle3 submission; parameters at most2x.

Read [Chinese analysis](analysis_report_zh.md), [protocol](protocol.md), frozen
[review](rml_review.md), and [expanded canonical appendix](rml_detailed_appendix.md).
The later catalog preserves172 pre-existing IDs plus2 FLAG IDs; the earlier
166-ID review remains an immutable CPU planning snapshot. It is not replaced.

CPU prospective `TB-k1-flag-cpu-qualification-20261002` closed NO_TRAIN as an
accepted engineering-only diagnostic. Exact gradient accumulation/next-step
resume and clean inference pass; no development/protected role read.
[Decision](cpu_decision.md), [attribution](cpu_attribution.md), independently bound
[acceptance](cpu_acceptance.json), and finalized RML remain separate from training.

Training prospective `TB-k1-flag-kaggle3-100k-s42-v1` is published under
`kaggle3_v1/flag/`. One random-initialized original K1 arm,3658817 parameters,
FLAG M3/alpha0.001, fixed100K/40epochs/31240 updates; no reference retraining.
Source commit `a4d85a520686af1574bfb025242a3c79783e6ddf`;
package `95dff6018e83142d3f773cc9fc6e6fe5bde09dac7cc81a761f10f774bb6e8bba`;
archive `89d0626619fc9efb0db557f15bf7558f1130188f0c1ff2af3cab02c29f973b9d`.
Private source dataset `nvoid912/molgap-k1-flag-source-s42-v1` was independently
downloaded: all9 file hashes and mounted shape match the prepared package.

Actual submission: [kernel](https://www.kaggle.com/code/nvoid912/molgap-k1-flag-100k-s42-v1),
ID136744623,version1,account nvoid912, machineShape NvidiaTeslaT4. Authenticated
latest observation at2026-10-02T06:42:15.233397+00:00 returned RUNNING; actual entry
bytes and dataset/version/owner identity match. This is a platform observation,
not proof that GPU qualification or formal training started. No outputs/log
were available then; runtime certificates, actual phase, native cost and score
remain pending. Receipt is [launch_receipt.json](launch_receipt.json); raw
submission/observation/source verification and release bindings are retained.

Next action is exact-attempt reconciliation on user return. Retrieve family
manifest, runtime/architecture qualification, predictions, selected/resume
weights, complete trace and costs via existing Kaggle adapter; verify identities
and separate infrastructure/qualification/science. Do not retry from stale state.
Preserve the4x step veto and3meV/paired-positive-bound endpoint gate. Actual
runtime/reference mismatches exclude strict claims; no automatic500K/full or
official/test evaluation, model adoption, heartbeat or machine shutdown.

Preparation: `platforms/_records/kaggle/staging/k1_flag_workflow_v1/` (ignored,
small release/workflow records copied to `platform_preparation/`; source mounted
privately on Kaggle). Qualified initial and CPU-resume binaries stay ignored
under `qualification/`, exact hashes are in finalized evidence. No pytest suite
was added/run; executed CPU qualification is separately recorded above.
