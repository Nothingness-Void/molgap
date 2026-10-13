# Local input pipeline protocol - 2026-10-13

User authority: local tests of the fixed-index/block/bucket proposal.
Owner: desktop. This is a new execution diagnostic, not an accuracy screen.

## Reuse and evidence review

The shared packed graph loader already implements fixed indexing, shard SHA
validation, source_idx identity, mmap, and topology-only views. Do not build a
second SMILES/graph cache. Reuse `load_training_data`, `PassBatchSampler`,
the shared worker options, `FamilyKernel` for K1 and the existing EdgeState
GPS constructor plus normalized Gap update for the GPS9 control.

Prior evidence: [A100 diagnostic attribution](../pcqm_k1_colab_execution_profile/attribution.md)
found small loader waits on4096 in-memory graphs, not a full500K traversal.
The legacy PairGPS size bucket helper fixes companion groups across passes;
the new diagnostic uses randomized windows and retains deterministic resume.
Neither evidence answers fixed-index100K bucket throughput on local RTX5060.

## Frozen scope

- Accepted PCQM100K training role only, source_idx0..99999. Existing cached
  geometry fields are stripped by the shared loader; no geometry constructed.
- Do not load shard2/development, official validation, test or other roles.
- Global Python epoch shuffle is the baseline. All policies first choose the
  same retained membership and discard the same single global tail.
- Block shuffle uses2560-index blocks; shuffle block order and contents.
- Bucket shuffle uses randomized2560-sample windows, sorts each window by
  node count, builds physical BS128 batches and shuffles batch order.
- CPU: complete training traversals at workers0 and1; two traversals per
  strategy, same pass0 order, with a fresh lazy graph cache per strategy.
- GPU: fixed8192 training members chosen from the baseline order. Same member
  set for every policy; consume all64 BS128 batches after eight discarded
  warmup updates and fresh model/optimizer/RNG reconstruction. Three repeated
  timings with randomized policy order, for K1 and EdgeState GPS9-192/9.
- FP32, TF32 off, seed42, single supervised Gap L1, native AdamW4e-4/1e-5,
  clip1. No EMA, schedule advance, validation, model selection or retained
  trained weights. These are scratch update probes, not scientific endpoints.
- K1 uses the existing pinned seed42 initialization and accelerated layout.
  GPS9 is an existing matched192-hidden/9-layer scratch control, not the
  accepted full-production checkpoint or a new registered training family.
- Total worker wall at most1200s; GPU phase at most600s. No remote submission
  or automatic continuation. Leave existing desktop applications/processes alone.

## Measurement and decision

Record full CPU traversal wall/startup, padding slots, exact membership/order
hashes, GPU end-to-end throughput, loader wait, CUDA step-span and peak allocated
VRAM. CUDA spans include transfer/update and host submission gaps, not exclusive
kernel time. Do not sum overlapping timings into an invented compute percentage.
SHA validation warms disk pages, so first traversal is not a cold-disk benchmark.

Changes to sampling affect BN/gradients, even with unchanged hyperparameters.
No MAE, precision non-inferiority, GPU full-epoch time or T4/A100 extrapolation
is established here. A throughput signal can nominate a separately authorized
paired100K quality experiment; it cannot change the live sampler or running jobs.
No numeric promotion gate is created by this diagnostic.
