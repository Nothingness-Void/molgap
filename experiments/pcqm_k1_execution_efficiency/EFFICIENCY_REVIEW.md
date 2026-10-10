# K1 execution review

This question owns execution diagnostics, not architecture or model promotion.
Start with [CPU scope](analysis_zh.md), [exact CUDA layout](gpu_attempt_001/terminal_decision.md)
and [TF32 qualification/limits](tf32_attempt_001/terminal_decision.md).

## Actual path

OGB atom/bond embeddings + RWSE16 -> nine persistent-edge updates and local
ResGatedGraphConv/MLP/BN blocks -> single-slot mixers after layers3/6/9 -> mean
pool -> normalized Gap head. Parameters3658817; the trained checkpoint and
scratch initialization have different purposes and are never mixed as weights.
`DynamicEdgeGPSWrapper`'s GINE path is not the active K1 path.

Reusable exact addon: `k1_execution_layout.cpu_layout_context`. It computes dense
integer indices/masks from CPU PyG ptr once and shares them across mixers, with
explicit pooling size. No added parameters/buffers/state_dict keys. Use around the
existing family step, not a replacement model or training loop. No default enable.

## Dispositions and remaining candidates

[Host-path audit](HOST_OVERHEAD.md) owns repeated sampler construction and growing
history-publication findings. Its CPU metadata cache is optional infrastructure,
not a native training-speed or quality certificate.
[JSON decomposition](JSON_PUBLICATION.md) identifies encoding as the larger
synthetic long-log cost and qualifies compact formatting through the existing
atomic owner; no records or checkpoints are removed.

| Candidate | Code owner | Minimum discriminator and risk |
|---|---|---|
| Project nodes before gathering endpoints | `gps.py::_PersistentEdgeUpdate.forward` | [Rejected as a strict-FP32 replacement](projection_attempt_001/terminal_decision.md): nonexact scratch state and mixed-sign timing windows. Algebraic equality is not bitwise equality. |
| Project real mixer nodes before packing | `qm9_neural_atom.py::NeuralAtomMixer.compute_update` | [Rejected as a strict-FP32 replacement](projection_attempt_001/terminal_decision.md): nonexact and slower than its matched layout control. |
| Scalar-rich checkpoint snapshot copy | `checkpoint_snapshot.py::checkpoint_to_cpu` | [Locally exact CPU helper](checkpoint_io_attempt_001/terminal_decision.md); synthetic history sizing only. Native sender throughput and end-to-end benefit remain unqualified. No active sender patch. |
| Foreach/fused optimizer | `pcqm_500k_v4_evidence.py::optimizer_for` | Previous local CUDA probe is much faster but nonexact. Needs explicit optimizer-contract and quality qualification, not silently enabling defaults. |
| Compile deterministic local subregions | `gps.py::_PersistentEdgeUpdate`, local blocks | Only after launch attribution. Compile startup, shape specialization, RNG/dropout/BN and Windows support are unqualified. Whole-model compile/CUDA graphs are not ready. |
| Host metrics/guards/RNG batching | family step/worker | Finite checks, loss extraction and RNG capture remain synchronization sources. Do not remove durability or finite guards; changing accumulation/check frequency needs its own execution semantics. |

K=1 slot attention is not an identity: value/output projection and dropout still
affect values and RNG. Deleting it is architecture/training research, not free
execution optimization. More GPU capacity does not automatically accelerate many
small kernels or CPU launches; do not infer A100/T4 equality from different runs.

## What TF32 means here

The helper temporarily permits lower-mantissa FP32 matmuls, with FP32 storage and
accumulation, not FP16 AMP. It restores backend settings even on exceptions.
The original Windows profiler lacks hardware dispatch evidence. The separately
[accepted CUDA trace](kernel_attempt_001/terminal_decision.md) now establishes
TensorOp-name GEMM dispatch, not utilization counters or native throughput.
[PyTorch2.7 CUDA semantics](https://docs.pytorch.org/docs/2.7/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
describes the speed/rounding tradeoff and hardware dependence; its large-A100-GEMM
example is not a K1 speed prediction.

T4 is Turing and does not provide TF32 Tensor Cores. The precision probe rejects
CUDA capability below8 (or unavailable CUDA); a permissive backend flag is not
hardware support. Native TF32 qualification means A100/Ampere-or-later, not T4.
T4 Tensor Core use would require a separately validated FP16/mixed-precision
contract, not enabling TF32. Sources: [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
and the PyTorch CUDA semantics linked above.

Small frozen-output perturbations do not qualify retrained MAE. Local same-seed
short-step comparisons are not independent training seeds or native hardware
performance certificates. No new remote job or training is released by this review.

## Practical qualification order

[Joint measured result](combined_attempt_001/terminal_decision.md): matched fused
increment26.54%, TF32-on-fused increment5.60%, combined original-to-candidate32.34%
step-time reduction. All paired windows positive for these contrasts; no retrained
MAE or native epoch qualification. Layout-only windows are mixed in this attempt,
so earlier about7% results are not a universal guarantee.

1. Conservative execution candidate: existing CPU layout. Short CUDA state/RNG
   and resumed continuation are exact; independent local attempts show about7%
   step-time reduction. Native loader/epoch/storage timing remains unqualified.
2. Larger speed candidate: foreach/fused AdamW. Their local22.79%/25.12% time
   reduction exceeds the isolated TF32 gain, but rounding and optimizer contract
   differ. Do not infer quality preservation from smaller scratch tensor errors.
3. TF32 is a precision candidate for Ampere-or-later, not a T4 optimization.
   The matched-layout increment is6.23%; frozen TRAIN prediction drift is small,
   but retrained development MAE is unknown. Reuse `precision_case` rather than
   adding another training loop. Any quality comparison needs separate authority.
4. Scalar snapshot helper is locally exact on CPU and helps long metadata
   fixtures. Full-history serialization/write cost remains; its24.57% largest
   fixture publication reduction is not a training-step saving.

Do not add these percentages: they measure different scopes/windows. Neither
projection reorder is a direct replacement. No default or active source changed.

## Local compile capability observation

2026-10-10, required project Python, CPU metadata-only command:

```python
import importlib.util, platform, torch
platform.system()                         # Windows
torch.__version__                         # 2.7.1+cu128
callable(getattr(torch, "compile", None))  # True
importlib.util.find_spec("triton")         # None
torch.cuda.is_initialized()               # False
```

The compile API exists but Triton is not installed in this environment. A CUDA
Inductor/Triton fusion test is therefore not qualified here; no compile/model
execution or dependency installation occurred. This is a local capability gap,
not proof that all torch.compile backends or a Linux native runtime are unusable.
Do not spend a GPU run on this path without first preparing a compatible backend
and separately budgeting startup, RNG, BN, exactness and steady-state checks.
