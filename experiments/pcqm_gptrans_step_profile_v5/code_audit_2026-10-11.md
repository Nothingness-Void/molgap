# GPTrans training execution audit — 2026-10-11

## Conclusion and scope

The inspected GPTrans routes contain redundant tensor preparation, synchronous
development transfers and fragmented eager execution. This supports engineering
work; it does not establish a 25% speedup or authorize changing the scientific
recipe. No production/trainer/model code was changed, no model was instantiated
or executed, and no remote job was accessed or submitted in this audit.

The audit followed the released author-screen dispatcher into
`src/molgap/pcqm_gptrans_v4.py` (100K local/triplet),
`src/molgap/gptrans_scale_ema.py` (equal-update 500K), and the historical
`src/molgap/pcqm_gptrans_prenorm_500k.py` route. Model/input owners reviewed were
`gptrans.py`, `gptrans_capacity.py`, `gptrans_triplet_communication.py`,
`gptrans_author_variants.py`, and `gptrans_author_inputs.py`.
The existing output hooks were checked separately from the actual author-screen
call path. All eleven inspected Python files passed AST parsing; this is not a
runtime equivalence test.

## Retained measurements, not new profiling

The accepted local500K run's four-update T4 Chrome trace was read as JSON. Its
SHA256 matched `completion_manifest.json`:
`65915deb063d07d734b5ed93de1f06d55f6afd5e0b606e02165af12924524862`.
Evidence root:
`platforms/_records/kaggle/training/gptrans_local_transfer500k_v1/gptrans_local_transfer500k/scale_ema/training/`.

- 35,720 CUDA kernel events / launch calls in four recorded updates.
- 12,132 launches were inside four CPU AdamW scopes: 33.964% of launch count,
  **not** 33.964% of training time. CPU/GPU annotation duplicates were excluded
  when counting optimizer scopes.
- 312 `aten::contiguous`, 872 `aten::clone`, and 2,588 `aten::copy_` events.
  Contiguous copies were enclosed by indexed writes, embedding backward and
  softmax backward; the evidence does not specifically blame Conv2d layout copies.
- Four `aten::equal` calls correspond to the per-forward dense-mask check.
- Profiling covered the first updates and included first-use allocation and
  profiler overhead. Nested CPU durations overlap; they must not be summed into
  utilization or a steady-state wall-time attribution.

Across the 59 non-profiled rungs (`timing/rung_01.json` through `rung_59.json`):

| Recorded region | Total seconds | Mean per rung |
|---|---:|---:|
| Training, including loader and H2D | 9841.41817744 | 166.80369792 |
| Three development evaluations | 4180.54508103 | 70.85669629 |

Evaluation was 29.8143% of **training plus evaluation**, not of complete job
wall time. Publication/checkpoint work after these timers is excluded. The
training timer itself has no explicit terminal CUDA synchronization, so it is
not an isolated GPU execution timer; following evaluations can absorb pending
work. These aggregate observations identify a substantial evaluation region,
not the avoidable portion of it.

The older [Kunshan full-step decision](results/decision_122499114.md) measured
forward/backward at about 87% of synchronized step time, H2D below 0.3%.
That was another runtime/model route, not a substitute for T4 local/triplet
operator attribution. Its foreach/finite-check variants did not pass frozen
trajectory equivalence. The
[shortest-path decision](../pcqm_gptrans_shortest_path_profile_v5/results/decision_122484011.md)
also failed speed/tail/equivalence gates; do not reopen it as an established fix.

## Findings

| Finding and source | Repetition | Interpretation |
|---|---|---|
| `gptrans.py:234–238`: dense categorical nodes are produced only to derive shape/mask, followed by another `to_dense_batch` for embedded nodes | Every forward, both training and evaluation | Duplicate scatter/index preparation. The local installed PyG implementation also reads GPU scalar batch/node maxima during each conversion. Runtime PyG bytes were not independently pinned by this static inspection. |
| `gptrans.py:195–200`, `gptrans_capacity.py:70–72`: graph-local edge indices are derived inside dense-input preparation and again by the local wrapper | Twice per local/triplet forward | Repeated batch maximum, dynamic `torch.where`, position/index construction; candidates for one shared per-batch layout. |
| `pcqm_gptrans_v4.py:244–262`: training loader rebuilt each epoch; evaluation uses workers0 and no pinned memory | Every epoch/evaluation | `persistent_workers=True` cannot preserve workers across discarded loader objects. `non_blocking=True` alone does not establish CPU/GPU overlap for unpinned evaluation batches. |
| `pcqm_gptrans_v4.py:511–550`: predictions, targets and row IDs transferred to CPU in each development batch | 391 batches per 50K pass | Prediction `.cpu()` is blocking; targets/IDs originate on CPU, travel to GPU with the entire PyG batch, then return. EMA/live passes repeat collation and transfers. |
| `gptrans_scale_ema.py:112–180`: two EMA updates per step and live/EMA9999/EMA999 evaluations each rung | Every update/rung | Three weight views are intentional observations, not three accidentally trained encoders. Preserve all declared observations; optimize shared input staging and transfers instead of deleting passes. |
| `pcqm_gptrans_v4.py:418–434, 469–509`: per-tensor eager EMA plus AdamW `foreach=False` | Every update | Many small operations and memory passes; the T4 trace confirms launch fragmentation. Backend changes require prospectively declared numerical qualification. |
| `gptrans.py:65–84`: dense node/pair projections, softmax, masks and reductions in 12 blocks | Every forward/backward | Genuine learned work plus eager launch/memory overhead. Flash/SDPA is not a drop-in replacement because attention/logits also produce pair updates. |
| `gptrans_triplet_communication.py:59–118`: pair masking/projections and inward/outward contractions at four insertions | Every triplet forward | Additional learned work. Aggregation avoids a cubic-size score tensor but its contracted arithmetic is still cubic in node count; attention materializes cubic scores. Few extra parameters do not imply negligible runtime. |

The legacy PreNorm500K runner repeats loader/evaluation/optimizer orchestration;
the contemporary scale route already reuses the native model/step/evaluation
owners. Repeated source code is a maintenance risk, not evidence that both
runners execute on each update.

## Checks that did not support a claimed bottleneck

- Shards are loaded into in-memory datasets before the epoch loop; no per-batch
  `torch.load` of shards/checkpoints was found in these training paths.
- Path sidecars load compact tensors once. Item access slices/clones RAM tensors;
  it does not reread the full CSR file. Path modes add repeated GPU content checks,
  but the retained local/triplet models do not use that path addon.
- Training losses accumulate on-device and convert once at the epoch/rung
  boundary; the current100K loop does not call `loss.item()` every batch.
- Current100K finite checks occur every50 steps; scale/legacy checks occur every
  step. Do not claim current100K still pays the old per-step finite-check cost.
- RML diagnostics, checkpoint hashes and atomic saves occur at boundaries, not
  each batch. The optional family-output hook can duplicate checkpoint/selected
  publication when enabled, but the inspected author-screen dispatcher does not
  pass `family_outputs`; it is not evidence of duplicate writes in the latest run.
- The100K row's elapsed timer is captured before native checkpoint publication,
  and its cumulative field sums those rows. It is not complete job wall time.
  Do not infer cheap checkpointing from those rows; use allocation/job costs or
  separately timed publication regions. Durability and RML fields must not be
  removed to improve a reported epoch time.
- PyG `.to()` can transfer unused geometry/RWSE/metadata, but retained measurements
  do not establish that these bytes dominate runtime.

## Recommended engineering order, not a compute release

1. Preserve identical rows/order/BS128/FP32/RNG and all live/EMA observations.
   Start with one shared dense layout/local-edge derivation per batch, and
   development input staging with CPU-owned targets/IDs. Buffer predictions and
   perform the same CPU metric reduction after one end-of-pass synchronization.
2. Make worker persistence real across epochs only after sampler order, worker
   RNG, transforms and resume behavior are verified. Do not change graph-size
   grouping or sample order to obtain a flattering speed number.
3. Qualify per-tensor EMA/AdamW execution alternatives and layout/kernel fusion
   separately. Existing foreach profiling is available; do not implement a new
   optimizer or silently select a previously failed backend. CPU-resident Adam
   scalar state is not automatically a CUDA wait; `aten::item` count alone cannot
   diagnose GPU synchronization.
4. A later authorized benchmark should separate loader, H2D, dense preparation,
   learned forward/backward, optimizer/EMA, live/EMA evaluation, publication and
   checkpoint costs on representative and size-tail batches. It must compare
   predictions, gradients, model/optimizer/EMA state, RNG, selected metrics and
   complete resume semantics before any execution change is admitted.

This audit did not select or implement an optimization, change a contract,
create a scientific RML trajectory, run training/inference, or access protected
roles. It found actionable sites; the speedup remains unmeasured.
