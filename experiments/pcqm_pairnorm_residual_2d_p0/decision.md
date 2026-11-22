# S1-P0 passes on development-fit residuals, not on training-fit residuals

2026-11-22, Track B. Close this local diagnostic as `NO_TRAIN` (no backbone
training). The S1-P0 verdict is **PASS**, not model adoption. Do not fit the
tested correction on the existing backbone's training rows for deployment.
No H1, remote run, protected-role evaluation or Track A change is released.

## S1: same checkpoints, fit sizes and scoring rows

MAE in meV; delta is chemistry minus the yhat-only control, negative is better.
The two fit roles each supplied 30,155 rows. All four corrections predicted the
**same 19,845 development molecules** on scaffolds absent from either fit.
The scalar control was chosen without scoring labels, inside each fit set.

| Init | Correction fitted on | Raw | Yhat-only (control) | Chemistry | Delta [paired 95% interval] |
|---|---|---:|---:|---:|---:|
| s43 | Backbone-unseen dev rows | 100.48 | 100.36 (affine) | 96.36 | -3.99 [-4.47, -3.53] |
| s44 | Backbone-unseen dev rows | 99.93 | 99.93 (median) | 95.81 | -4.12 [-4.63, -3.65] |
| s43 | Backbone training rows | 100.48 | 100.30 (affine) | 101.14 | +0.84 [+0.65, +1.03] |
| s44 | Backbone training rows | 99.93 | 99.91 (affine) | 100.67 | +0.77 [+0.62, +0.92] |

Both primary deltas clear the predeclared -2.0 meV gate. The train-fit
correction is worse than both its scalar control and the unmodified model.
Thus chemistry predicts some residual error in this 500K regime, but how the
correction is fitted changes the practical answer. Matching the checkpoint,
fit count and scoring cohort rules out those three differences as explanations.
It does **not** separate training membership from differences in the molecules
and scaffolds available in the two fit sources.

The intervals sample paired rows with models and partition fixed. The 19,845
scoring rows have 4,094 scaffold keys; these are not scaffold-bootstrap or
training-seed intervals. All acyclics share the empty scaffold (fit side).
Neither init saw these development labels as backbone training targets, but
both used the 50K development role for checkpoint selection. No full-data
or independent evaluation was performed. H1 is worth **considering** only
with separate authority and a design that measures net gain against a
same-init full-training-data baseline, including the cost of withholding rows.

## N4: one predeclared group clears all three gates

On all 50,000 dev rows the raw MAEs are 101.09 / 101.32 meV, reproducing
the parent endpoints. Only **no pi bond** (RDKit: all bonds single) qualifies:
4,312 rows (8.62%), 158.44 / 162.80 meV MAE, 1.57x / 1.61x the global MAE,
and 13.52% / 13.86% of total absolute error. On 28.90% of these rows both
inits miss by at least 150 meV in the same direction, versus 15.69% overall
(1.84x). Acyclic molecules also have high MAE but contribute only 4.13% /
4.45% of absolute error. Groups overlap. N4 measured raw errors: it does not
show which group the stacker fixed or nominate a proven training intervention.

## Execution and limits

The [protocol](protocol.md) and [analyzer](probe.py) were frozen at `094f1b57e`;
the [prospective trajectory](rml/trajectory.json) was committed at `ea58fa71b`
before reading experiment rows. The parent passed local read-only strict
acceptance. Each frozen RTX 5060 replay matched the first 128 saved dev
predictions to within 0.0000008 eV before predicting training rows. The
scoring and N4 inputs were the saved dev predictions, not newly selected
checkpoints. One unparseable dev molecule stayed in the cohort with missing
chemistry. No rows were dropped for parse or descriptor failures.

Executed with the primary checkout's project Python, `PYTHONPATH` set to this
worktree's `src`, `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`. Stage order in `probe.py`: `features`, `replay`, `s43-dev`,
`s44-dev`, `s43-train`, `s44-train`, `analyze`. Only the last stage scored.
Summed in-stage wall time was 414 s, including 38 s for the local replay stage;
the latter includes CPU and I/O, not measured GPU-hours. Total CPU-hours are
unknown because feature-worker CPU time was not recorded.

[Analysis and stage hashes](analysis.json) own the unrounded results.
Per-row features, split membership, replay and the four predictions are in
`D:/molgap-artifacts/diagnostics/pcqm-pairnorm-residual-2d-p0/`. Read the
saved artifacts rather than refit or score a second time.
