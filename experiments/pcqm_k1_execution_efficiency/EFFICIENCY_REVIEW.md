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

## Ranked next changes

| Candidate | Code owner | Minimum discriminator and risk |
|---|---|---|
| Project nodes before gathering endpoints | `gps.py::_PersistentEdgeUpdate.forward` | Each layer separately projects gathered source/target edge rows. Projecting all real nodes once per projection and then gathering may reduce repeated work; compute advantage depends on edges/nodes. GEMM shape and backward reduction order change: synthetic full-gradient check first, then prospective scratch state/RNG test. Algebraic equality is not bitwise equality. |
| Project real mixer nodes before packing | `qm9_neural_atom.py::NeuralAtomMixer.compute_update` | Bias-free node key/value projections currently include padded nodes. Project-before-pack can reduce padded work; first measure occupancy. Uneven-size graph forward, assignment and gradient checks must qualify changed GEMM/reduction ordering. |
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
Hardware dispatch is an explicit evidence gap in the Windows profiler result.
[PyTorch2.7 CUDA semantics](https://docs.pytorch.org/docs/2.7/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
describes the speed/rounding tradeoff and hardware dependence; its large-A100-GEMM
example is not a K1 speed prediction.

Small frozen-output perturbations do not qualify retrained MAE. Local same-seed
short-step comparisons are not independent training seeds or native hardware
performance certificates. No new remote job or training is released by this review.
