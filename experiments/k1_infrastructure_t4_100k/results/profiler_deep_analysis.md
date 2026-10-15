# K1 profiler: code-level opportunities — 2026-10-15

This saved-trace analysis supplements the immutable [terminal result](decision_2026-10-15.md).
[Machine evidence](profiler_deep_analysis.json) binds both original Chrome traces,
the extraction script and full attribution. Eight inspected runtime files matched
the frozen `eb640c96` package. No model execution, training, new GPU job, recipe
change or claimed additional speedup occurred.

## What the capture establishes

The optimized capture contains39,136kernels:2,446per update. Median kernel
duration was8.991us;52.45% lasted under10us. Captured leaf device activity totals
974.680ms in3.435s wall. Its28.38% capture coverage is **not unprofiled GPU
utilization**: instrumentation, host gaps and missing hardware telemetry limit it.
Leaf GPU activity was counted once through External-ID correlation; overlapping
intervals were merged. Operator summaries, kernels and GPU annotation ranges
must not be added together. Two device events (32.096us) lacked a matching CPU
operator; autograd-thread events were matched to enclosing backward ranges.

| Optimized capture attribution | Device duration | Share of leaf activity sum |
|---|---:|---:|
| All forward `scatter_add_` descendants | 350.169ms | 35.93% |
| Nine local graph aggregations within that row | 335.828ms | 34.46% |
| `mm` + `addmm`, forward and backward | 274.749ms | 28.19% |
| All `_index_put_impl_` descendants | 298.026ms | 30.58% |

The last row overlaps the first: it is not an additional30.58%. Of index-put
activity, at least255.114ms was identified as sorting/indexing kernels inside
**forward scatter aggregation**; corresponding backward index-select and
advanced-index sorting chains contributed24.332ms and14.794ms. The earlier
operator ranking alone did not establish that backward was the main cause.

The optimized forward had11scatter calls per update: nine ResGatedGraphConv
aggregations, then graph-mean count/value reductions. The original had14,
including three dense-layout count reductions already removed by CPU layout.
All160remaining `cudaStreamSynchronize` calls descended from a CPU stride tensor
copy to CUDA inside two-dimensional deterministic scatter. These consumed15.227ms
of host time in capture; their count alone does not explain the whole wall gap.

AdamW accounted for10,016of10,112`item()` calls, exactly2x313parameter states
x16updates. They read CPU step scalars for foreach bias correction and had no
CUDA synchronization descendants. Their inclusive host time was12.244ms.
The retained trace already contains foreach AdamW operations; enabling foreach
is not a new optimization. [Pinned PyTorch optimizer source](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py)
explains the CPU scalar path. Optimizer host range was151.008ms; scalar reads
are a modest part of it.

## Ranked implementation opportunities

| Priority | Concrete change and owner | Evidence / qualification |
|---|---|---|
| Largest measured opportunity | Cache a stable destination permutation and segment pointers once per CPU batch; use a deterministic segmented aggregation provider for the nine local blocks. Owners: `k1_execution_layout.py`, registered execution adapter and local aggregation hook. | Targets repeated coordinate construction/sorting in350.169ms of captured forward activity. Native output/gradient/optimizer/RNG/resume equivalence and unprofiled timing remain required. No gain estimate is established. |
| Smallest first code change | Apply the already admitted memory-fill scope consistently to K1 evaluation forward in `family_training_kernel.py:forward`. | `step` disables safety fill, but `forward` lacks that scope. Eval has no capture here, so saved work and numerical equivalence are unmeasured. Preserve deterministic algorithms and verify no uninitialized storage reads. |
| Small infrastructure change | Validate one immutable checkpoint snapshot, combining device finite reductions before host observation or moving complete finite checks to the CPU snapshot. Owners: `family_training_session.py`, `experiment_family_workflow.py`, `family_training_outputs.py`. | `state_dict` scans the full state per tensor, then `write_resume_state` scans model tensors again. The endpoint has367model tensors and626optimizer moment tensors. Keep full checks, atomic publication and error diagnostics; checkpoint time was not captured. |
| Second execution change | Reduce repeated endpoint gathers/projections in `_PersistentEdgeUpdate.forward` in `gps.py`. | Two192-to64projections currently run on gathered edge rows. Projecting nodes first can reduce work when E>N; parameters need not change. Floating-point gradient accumulation order changes, so mathematical identity does not guarantee bitwise training identity. |
| Later, targeted compilation | Compile admitted tensor-only blocks, isolating CPU topology preparation, IO and observers. | Many tiny kernels and dispatch costs support investigating it. Dynamic node/edge counts, BN buffers, dropout RNG, deterministic scatter and resumes complicate capture. Compilation cost must be included and gains measured. |
| Low priority | Further dense pack/unpack, layout worker offload and pinned topology buffers. | Loader wait was2.823ms, layout-enter20.304ms and transfer8.060ms of3.435s capture; layout-generated tensors are newly allocated CPU tensors despite pinned model inputs. These are small measured scopes and a new process/pinning step may cost more. |

## Preserving the contract

PyTorch2.7.1 deliberately routes deterministic CUDA scatter through
`_scatter_via_index_put`: it expands coordinates across feature channels,
copies stride metadata and invokes sorted indexing. This is the source-level
explanation of the observed trace, not unnecessary model arithmetic.
[Pinned implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/TensorAdvancedIndexing.cpp).
PyG2.6.1 can use a segment path when an aggregation receives pointers;
[the aggregation owner](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/aggr/base.html)
already exposes that seam. This is a candidate integration, not a qualified
replacement for this experiment's scatter order.

Sort only the **aggregation view**, keeping canonical edges, edge features,
edge-state alignment and dropout assignment unchanged. Sorting the whole graph
before edge updates reassigns dropout draws to bonds. Preserve duplicates and
isolated-node/empty-segment semantics. Stable input order still does not promise
the same floating-point reduction tree; require exact native checks if bitwise
compatibility remains the acceptance criterion. Never disable deterministic
algorithms merely to obtain an atomic scatter benchmark.

Keep topology metadata batch-local and outside model state. Use an execution
provider/hook; do not fork the model, introduce an experimental addon for a
transport change, or install a global PyG patch. Snapshot validation belongs
to the shared persistence owner and should benefit all admitted families.

Do not start with whole-model CUDA Graphs: the pinned implementation has dynamic
shapes and CPU transfers/synchronization. PyTorch's [version-matched guide](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/torch.compiler_cudagraph_trees.rst)
documents per-shape recording and CPU-operation restrictions. Fused AdamW,
capturable mode, TF32/AMP, changing graph order/batching, removing dropout or
shortening development exposure are not transparent fixes under the frozen
contract. The three one-slot attention blocks are not the dominant measured
cost; no attention replacement is justified by this trace alone.

The recommended implementation order is evaluation scope and checkpoint
validation, then aggregation, then projection/targeted compilation. A bounded
native check should cover diverse shapes, duplicates/empty edges, profiled/plain
and serialized-resume state, plus a pass boundary; report train, eval and save
separately. This analysis grants no new remote execution or full100K rerun.
