# Input-initialization failure-mode attribution — 2026-09-30

The completed pair passed its frozen source, data, runtime, artifact and role
checks. Both arms used the same Kaggle1 kernel, accepted graph manifest,
100K training rows, 50K internal-development rows, seed, optimizer, schedule,
FP32 recipe, EMA selection and 60-epoch exposure. The candidate's complete
initial-state hash reproduced the Kaggle1-pinned value. The failed first
attempt was an infrastructure hash mismatch; it supplied no training evidence.

At the accepted endpoint the candidate is 1.269 meV worse than the same-job
control, with a paired row-bootstrap interval wholly above zero. The point
effect is opposite to the required 3 meV gain. This supports a negative result
for the 15-table input-scale intervention under the frozen 100K contract.
Parameter count is identical, and the alternating same-device preflight passed
both latency ratios. The full 60-epoch interval sums differ by about 0.0010
single-T4 hours; these timings do not establish a general speedup.

The candidate initially had lower development MAE, but its development MAE was
higher than the control's at every epoch from epoch 26 through 59. Both selected
their best EMA development checkpoint at epoch 59 and both absolute development
curves were still improving. The final online training-batch MAEs were 0.103554
and 0.103974 eV for candidate and reference; these are live, changing-batch
training measurements, while development uses EMA on a fixed cohort. Their
absolute difference cannot diagnose overfitting or underfitting. The retained
traces show the relative candidate deficit stayed between 1.269 and 1.656 meV
over epochs 50–59, without proving a convergence plateau or excluding a later
reversal beyond the authorized exposure.

Paired source indices and targets matched exactly; 48.752% of rows had a lower
absolute error under the candidate. The experiment did not predeclare
mechanistic slices, and no seed-variance estimate exists. Altered optimization
dynamics and adaptation to the project's atom/bond composition remain compatible
explanations; the evidence does not isolate their causes. A matched-exposure
continuation or additional seed would be the direct discriminator of a late
reversal or seed sensitivity, but neither is released by this negative result.
