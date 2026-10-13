# Execution Audit - 2026-10-13

Scope: K1 and GPTrans shared wheel, not archived model families. No model/loss,
precision, optimizer, sampling, learning-rate, BN/dropout or save-policy changes.
This audit does not amend V5 or native release gates.

## Existing Work To Reuse

| Path / owner | Existing behavior | Disposition |
|---|---|---|
| `packed_graph_dataset.py`, `family_training_data.py` | Stable source indices, prebuilt packed graphs, mmap and hash admission; inherited lazy PyG graph caching | Keep; no repeated SMILES/ETKDG in the training loop |
| `family_training_data.PassBatchSampler` | Frozen pass order and resumable batch cursor; private generator | Keep global sampler; previous block/bucket probe does not justify changing batches |
| `family_training_loaders.py` | Persistent input pool, selected role only, CPU layout before transfer | Keep; workers are config/resource policy, not automatically increased |
| `k1_execution_layout.py` | One validated CPU layout shared by three mixers; explicit graph count avoids GPU `batch.max()` for pooling | Already adopted; no new layout implementation |
| `gptrans_execution_backend.py` | Prepared layout, fast dense assembly and ordered buffered development output | Already adopted; no duplicate cache/trainer |
| `family_training_kernel.py` | Borrowed CPU storage and selective model-field transfer | Keep; geometry and source indices are not transferred merely because present |
| `family_training_core.optimizer_update` | Detached metrics, no per-step reporting copy; frozen zero-grad/backward/clip/step order | Keep native FP32/noTF32 and clipping semantics |
| `gptrans_execution_hotpath.py` | Bound foreach EMA lists, cached weight backup and schema/storage guards | Already adopted; do not remove mutation detection to shorten CPU guards |

For K1, native AdamW uses unspecified foreach/fused values. Modern Torch may
choose a backend automatically; it is wrong to attribute current cost to a
forced scalar optimizer using an old unfused profile. GPTrans explicitly uses
foreach. This change touches neither setting.

## Fixed Shared Overhead

`FamilyTrainingSession.state_dict()` checked the complete nested state via a
Python boolean per GPU tensor. The independent model/EMA writer checks already
batched their device transfers. The fix extracts that existing strategy into
`nonfinite_tensor_keys` and reuses it for the nested checkpoint inspector.

Every tensor is still checked. Accelerator flags cross to CPU once per device
per validation call; CPU scalar counters and pending metrics are checked in
same-dtype groups. Ordinal tensor keys preserve list/path collisions, original
dict traversal priority, host-float failures and exception labels. There is no
success cache: changing a tensor after a successful check is still detected.

## Work Deliberately Not Removed

| Work | Why it remains | Required next discriminator |
|---|---|---|
| Complete model/AdamW/EMA validation at persistence | Last checkpoint must reject nonfinite state independently of occasional loss checks | Measure matched state checking and atomic IO separately |
| Independent writer finite check | Direct writer callers must not trust a session summary boolean | Retain the safety boundary; optimize its shared checker |
| Partial-pass metric prefix snapshot | Exact resume depends on original scalar values and Python addition order | Private append-only metric-buffer design plus exact mutation/resume tests before caching |
| Checkpoint cadence | Interval128, pass boundaries and pause/final saves belong to the frozen recipe | Any cadence experiment needs new source/config and durability authority |
| Edge-endpoint projections at every layer | Project-before-gather changes GEMM shapes and backward reduction order | Separate numerical/gradient/optimizer qualification, then quality if not exact |
| Scatter/norm/linear kernels | Genuine forward/backward work; deterministic reductions are part of execution | Current native operator/stage profile, not an old A100 percentage |
| Development evaluation/EMA weight switching | Frozen selection/role policy, not dispensable monitoring | Separate applicable role and selection contract; never silently skip rows |

## Attribution Limits

The preceding [input diagnostic](../pcqm_input_pipeline/terminal_decision.md)
already excluded a large stable bucket benefit on its matched8192-row windows.
Its K1 baseline median loader wait was0.713ms versus52.951ms CUDA update span;
GPS9 was0.918ms versus84.542ms. The event span includes host submission gaps and
transfer/update, not exclusive kernel time. These are not full500K epoch costs.

The earlier [A100 profile](../pcqm_k1_colab_execution_profile/attribution.md)
supports examining forward/backward small-operator overhead, but predates the
adopted layout/runtime. Its exact phase percentages must not be applied to the
new wheel or used to declare a universal A100/T4 ratio.

This diagnostic measures only finite validation and a synthetic atomic-IO proxy.
It cannot prove total training speedup, final MAE, native T4/A100/DCU qualification
or that no further optimization exists. No new remote run or successor follows
automatically from the measurements.
