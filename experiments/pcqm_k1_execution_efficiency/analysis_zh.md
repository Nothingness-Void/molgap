# K1 execution-efficiency review

## What was tested

The existing K1 factory, normalized-L1 step, FP32/RNG/checkpoint helpers and retained
4096 training-member probe were reused. No new model/trainer/platform launcher.
No development/protected roles, scientific candidate fitting or remote submission.
The actual CPU probe uses torch2.7.1/PyG2.7.0, four intra-op threads, physical BS128;
it is not the frozen torch2.4.1/PyG2.6.1 T4 runtime.

Same initial tensors, same ordered graph batches and replayed dropout RNG. Every
case has three numerical-control updates and four balanced timing windows of
three warmup plus twelve measured updates. Cloning input batches, finite guards,
H2D/context and RNG capture are included; loader/development/EMA/publication are
not included. Input batches are already CPU-collated, not a full native loader.

## Observations

| Case | CPU seconds/step | Tensor/RNG exact | Disposition |
|---|---:|---|---|
| Original loop |0.50566|Yes|Control |
| Nonblocking transfer |0.49676|Yes|CPU no-op; not CUDA speed evidence |
| Remove redundant barrier |0.50984|Yes|CPU no-op; not CUDA speed evidence |
| Transfer + barrier |0.51255|Yes|No stable speed gain |
| CPU topology layout |0.49557|Yes|Correctness candidate; two positive/two negative timing pairs |
| Foreach AdamW |0.50619|Yes here|Optimizer-contract change; not adopted |
| Fused AdamW |0.49314|No|Max difference2.384e-7; exact gate fails |

The probe body measured236.30 seconds. Full worker wall/process CPU unavailable
because the tail publication metadata failed; preserve unknowns, not zero.
No percentage is claimed as accepted GPU saving. GPU load remained non-idle;
no accelerator scratch benchmark or other-process termination occurred.

## Prioritized next discriminators

1. CPU layout removes three repeated GPU topology-shape discoveries and boolean
   unpackings plus pooling shape discovery, without changing mathematical message
   flow. First qualify it on an idle GPU with the same strict state/RNG checks.
2. Retain finite guards and host loss; merely deleting explicit synchronize may
   have little effect because implicit synchronizations remain. Measure separately.
3. Foreach/fused are different execution contracts. Keep unfused defaults until
   independent native numerical/runtime qualification; do not credit tiny CPU timing.
4. Compile repeated local blocks only in a new bounded qualification: dynamic graph
   shapes, dropout RNG, changed reduction order and compile startup are obstacles.
   Windows native Triton support and pinned remote software must be checked first.
5. Record durable checkpoint publication separately from compute; never weaken
   publication/resume or extrapolate this one-model local save to two-arm remote IO.

The K=1 attention block cannot simply be deleted: value/output projections and
training dropout still matter even when the softmax has one entry. That would be
an architecture/training change, not an equivalent execution optimization.

## References and boundary

- [PyG compilation guidance](https://pytorch-geometric.readthedocs.io/en/latest/advanced/compile.html)
  identifies pooling size inference as a synchronization/graph-break source; its
  GIN benchmark is not a promised K1 speedup.
- [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  explains avoiding unnecessary device synchronizations.
- [Transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
  stresses environment-specific transfer behavior.
- [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
  describes foreach/fused implementations, not exact training interchangeability.

[Terminal decision](terminal_decision.md) owns disposition; [attribution](attribution.md)
owns limits. Corrected wrapper source is a later infrastructure revision; the CPU
attempt was executed from the retained `executed_source/` bytes, not the fixes.
