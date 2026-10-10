# Projection placement attribution

Both reorders retain model shape and state-dict keys but change GEMM shapes and
backward reduction paths. The controlled three-update comparison establishes
numerical non-equivalence, not generalization harm. Small gradient differences
can enter AdamW state; tolerant forward/unit checks cannot certify exact training.

Gather changes repeated endpoint-row projections into node projections plus
indexed reads. Matched barrier controls remove the known explicit-sync confound;
its mixed-sign timing windows do not establish a useful local speedup. Pack
projects only real nodes but introduces two narrow packs and guarding overhead.
The retained timings show no net benefit under this adapter; without valid CUDA
kernel profiling, the individual overhead contributions remain unmeasured.

The unchanged CPU-layout case stays exactly replayed and locally faster in four
windows, consistent with its earlier independent diagnostic. This supports that
bounded execution optimization, not a universal A100/T4 speed ratio or training
quality adoption. The trained-checkpoint and TF32 findings remain owned by their
separate prior attempts.

No weak-readout, underfitting, cross-scale transfer or dataset claim follows from
this execution-only probe. Further discriminators, if separately authorized:
native end-to-end layout timing; a working GPU profiler for kernel attribution;
matched quality qualification for any explicitly changed numerical contract.
Do not retrain a reference or consume protected roles to fill diagnostic gaps.
