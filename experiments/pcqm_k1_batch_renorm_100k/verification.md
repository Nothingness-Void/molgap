# Local adapter verification

2026-10-24. Reuse plan: K1/3 FamilyKernel, substrate replacements, original
checkpoint/EMA/RNG owners, scientific-input compiler and Kaggle submit_workflow.
New implementation is only k1_batch_renorm.py and its catalog declaration;
the shared training/receipt/RML framework is unchanged by this experiment.

Executed by Luna6 max with primary project Python, selected-checkout wrapper,
machine profile, CUDA_VISIBLE_DEVICES=-1, OMP/MKL threads1:

`tests/test_experiment_attribution.py::test_k1_batch_renorm_family_kernel_resume_journey`

First execution failed after12.60s because the test compared a3x2 tensor with
a2-vector using a non-broadcasting assertion. Parent fixed that assertion and
strengthened the gradient input so r/d are unsaturated: otherwise missing
stop-gradient would have escaped detection. Model code did not change.
The same node rerun passed in18.53s. No full-suite run or extra test files.

The test checks hand-derived detached input gradients, original unbiased running
variance updates, correction onset/caps, evaluation and cumulative calibration,
the18 actual PyG-wrapped local replacements, unchanged initial tensor keys and
parameter counts, real K1 FamilyKernel updates on synthetic graphs, serialized
model/optimizer/EMA/RNG restoration, exact next update and shared family model
reconstruction with strict selected-state loading.

Limits: synthetic CPU graphs and random fixture initialization, not native T4,
scientific rows or quality. Counter7809 is injected to cover enabled correction;
it is not claimed as7809 executed updates. Session cursor/scheduler are unchanged
and not retested here; native qualification remains mandatory for each arm.

Logs (D: external, no checkpoint fixture retained):
`D:/molgap-artifacts/reconciliation/k1-batch-renorm-100k/local_checks.log` and
`D:/molgap-artifacts/reconciliation/k1-batch-renorm-100k/local_checks_repaired.log`.
