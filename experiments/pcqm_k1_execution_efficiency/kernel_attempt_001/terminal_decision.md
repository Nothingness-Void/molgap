# Kernel attribution decision

2026-10-10; desktop NO_TRAIN. Source6641447b, bounded local diagnostic44.605s.
Authority: [observations](results/result.json), [FP32 trace](results/isolated_fp32.json),
[TF32 trace](results/isolated_tf32.json), [summary](kernel_summary.json).
101 focused CPU/regression tests passed before execution.

Two traced owner steps per successful case, after three discarded warmup steps;
same seed42, frozen initial state, first256 accepted TRAIN molecules, BS128,
CPU4threads, strict deterministic policy, unfused AdamW, CPU layout. Only the
matmul TF32 permission differs between the successful cases. No learned endpoint,
development/test consumption, remote job, recommendation or payload change.

Bundled CUPTI reports API26 and is rejected by this collector's verified API28
ABI boundary. This is unsupported collector ABI, not a newly reproduced device
failure. Separately loaded official12.9.79/API28 produced valid traces for both
precisions, with zero dropped records. No system/PyTorch DLL was replaced.
Because both library and tracing adapter differ from the historical Kineto path,
the new trace cannot isolate the cause of its historical INVALID_DEVICE error.

| Instrumented two-step observation | FP32 | TF32 allowed |
|---|---:|---:|
| Kernel records |13350|12938|
| GEMM calls |708|708|
| TensorOp-name GEMM calls |0|688|
| Summed GEMM duration |20.105ms|8.643ms|
| Indexing-backward duration |12.040ms|12.067ms|
| Radix-sort duration |4.642ms|4.638ms|
| Elementwise duration |14.747ms|14.965ms|
| Traced kernel interval union |63.570ms|50.500ms|

Groups overlap; do not add them as mutually exclusive category shares. All
recorded kernels are on device0/stream7. Kernel durations and trace-span gaps
are instrumented attribution, not throughput, long-run utilization or allocation
cost. GPU busy time for the whole worker remains unknown. Child process CPU
49.453s is measured separately; parent CPU, preparation/tests/closure excluded.

FP32 uses `cutlass_80_simt_sgemm_*`; the TF32 case uses
`cutlass_80_tensorop_s1688gemm_*`. Together with the controlled FP32 inputs and
matmul policy, these names support actual dispatch onto the TensorOp GEMM path,
not merely permission. [NVIDIA CUTLASS functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html)
and [SM80 MMA implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm80.h)
describe TensorOp/TF32 and the16x8x8 TF32 operation. This is dispatch-level evidence;
instruction counters, occupancy and Tensor Core utilization were not collected.

GEMM duration falls57.0% in this instrumented sample, but substantial indexing,
sorting and elementwise work remains. Do not replace the independently balanced
TF32 speed result (6.23% incremental step-time reduction) with this GEMM-only
percentage. Frozen4096 TRAIN prediction perturbation stays owned by the
[prior precision decision](../tf32_attempt_001/terminal_decision.md); retrained
and independent-role MAE remains unqualified. No strict-FP32 admission or new
scientific training is released by kernel dispatch evidence.
