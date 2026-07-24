# P8.17 Exact-2M GPS Distillation

## Hypothesis

Compress the accepted fixed equal ensemble of `control_a` and `repair_v2`
dual-GPS experts into one GPS7 student. The student is initialized from the
exact-2M GPS7 checkpoint and trained on the exact-2M graph cache with a fixed
loss of 70% teacher MAE plus 30% B3LYP-label MAE.

This is a development-only compression gate. The permanently sealed 10K and
future sealed 20K are not mounted or read.

## Durable Execution

- Teacher embedding task `703580_0`: first `control_a` GPS7 extraction.
- Teacher embedding array `703583`: remaining three 8-core/27-GB one-DCU
  encoders. The account also enforces `AssocGrpGRES`, so these array items run
  one at a time even though the payload permits two-way concurrency.
- Student task `703584`: starts only after both teacher dependencies succeed.
- Teacher embeddings are FP16 50K-row parts with an atomic manifest and
  per-part SHA256.
- Student training writes an atomic checkpoint and progress JSON every epoch.
- Final outputs include the student checkpoint, metrics, test predictions,
  complete 2M embedding parts, and an aligned 997,445-row FP16 embedding
  prefix for later SchNet fusion.

The initial 16-core array allocation exposed the account's association CPU
limit. Its three pending tasks and the original dependent student were
cancelled before they started; the running first task was retained, and the
remaining work was resubmitted using the validated 8-core/27-GB shape.
Task `703580_0` then completed all 2,000,000 rows in 14m10s with 40 durable
parts; `703583_1` started normally afterwards.

All four teacher extractions later completed. The first student attempt
`703584` built all 40 teacher-target parts, then exceeded its 27-GB allocation
while loading the 9.2-GB PyG object cache (measured peak about 29.3 GB). The
teacher work is intact and reusable; the student-only retry uses 16 cores and
54 GB after the teacher jobs have released the account allocation. Retry job
`703633` was submitted from the retained teacher targets.

The retry's first five epochs had its best label-validation MAE at epoch 0
(`0.10321`) and then regressed through epoch 4 (`0.10443`). A materially
different conservative follow-up, job `703653`, is queued with
`afterany:703633`. It keeps the same initialization and teachers but uses 30%
teacher loss, 70% B3LYP-label loss, LR `2e-5`, and a separate `student_gps7_w30`
run directory. The dependency prevents the two students from competing for the
same allocation.

## Decision Gate

First compare the single student with the fixed two-expert teacher on
development evaluations. Only a student that retains the teacher gain proceeds
to a newly trained student-embedding + 1M SchNet fusion head. Existing fusion
head weights cannot be reused because the 2D representation changed.
