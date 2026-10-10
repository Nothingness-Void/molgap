# Execution bottleneck attribution

Actual CUDA records fill the prior attribution gap: changing only matmul policy
in this collector moves GEMMs from SIMT-name to TensorOp-name kernels. Their
large isolated duration reduction does not remove the non-GEMM work. Indexing
backward, radix sort and thousands of elementwise launches remain almost unchanged
in the two-step sample. This supports mixed graph/reduction/launch bottlenecks,
not a claim that Tensor Cores were unused or that enabling TF32 solves K1 cost.

Each owner step dispatches roughly6500 kernels in this instrumented sample.
That makes host launch/synchronization overhead a plausible explanation for modest
whole-step gains. Kernel gaps cannot prove uninstrumented CPU bottleneck shares,
and these RTX5060 samples cannot establish A100-versus-T4 speed parity. Detailed
operator-to-kernel correlation, native hardware timing and real loader/checkpoint
cost remain discriminators. Never use tracer overhead as production cost savings.

The deterministic indexing/sort work is visible; this does not authorize turning
off determinism, changing reduction order, batch size or optimizer to improve
utilization. Prior projection reorders already failed exactness/cost gates and
stay rejected. Their scientific conclusions are not reopened by this trace.

The remaining precision question is training quality, not frozen output drift.
No independent/generalization outcome can be inferred from the trace, three
warmup updates or4096 TRAIN prediction perturbations. A matched quality run and
its reference require separate prospective resource/role/training authority.
No such run, native A100 test or successor was submitted here.
