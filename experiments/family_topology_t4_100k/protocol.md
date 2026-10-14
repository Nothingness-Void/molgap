# Execution Comparison - 2026-10-14

User authorization: resubmit the optimized T4 workload with profiler and compare
with the previous complete run. One Kaggle1 private T4x2 Notebook, one GPU per
arm, complete fixed100K contracts. No missing scientific reference is retrained.

Reuse exact frozen initial tensor files, data/cache identity, role membership,
row order, normalization, FP32/noTF32, BS128/drop-last, seed42, AdamW, schedule,
selection, checkpoints and profiler settings from the baseline. K1 mean2/live
completes40passes31240updates3998720presentations; GPTrans degree/bond-local
EMA.999 completes60passes46860updates5998080presentations. Profiler skips120
updates and captures16, including checkpoint128; shapes/stacks/memory off.
Train0..99999 and internal development100000..149999 derive from official train.
Official validation/test-dev/test-challenge are forbidden. This reused development
role is not an independent holdout and no geometry is constructed.

Only executable source changes: reuse K1 padding masks, reuse true-bond target
indices across GPTrans blocks and bound reachability by simple-path length.
No floating-point reorder, optimizer, dropout, schedule or sampling change.
Both arms need the existing source-bound native profiled/plain repeat and actual
serialized-resume barrier before formal training. Old certificates cannot qualify
the new source. Preserve atomic best/last state, predictions, trace, costs,
native proofs, profile report/trace and exact job/version identity.

Baseline is kernel138537758/version1, executable sourcefaf5b6d1, not the metadata-
only later integration HEAD. Freeze its retained artifact hashes before release.
Compare each family to its own baseline, never K1 to GPTrans as a causal pair.
Require matched recipe, input/initial-state hashes, complete exposure and runtime
before interpreting worker-wall differences. Report full worker-entry wall,
pair entry/allocation wall, profiler phases and operator counts separately.
Compare saved predictions in source_idx order, live/EMA and selected-pass meaning.
Profile windows include a checkpoint and do not cover evaluation; neither their
percentages nor asynchronous CPU/device rankings are whole-run exclusive costs.

This is a cross-job execution-delivery comparison, not STRICT_CAUSAL architecture
evidence or a new scientific material threshold. Queue, host contention, driver/
image drift and profiling overhead may explain small differences. If identities
or runtime differ, label the result contextual/inconclusive rather than forcing
comparability. Record regressions or no benefit without a new retry or full run.

Resource configuration retains25200s (7wall hours) including setup, with180s
cleanup reserve; at most14assigned T4-device hours over that envelope, not
platform billing. At the ceiling retain a checkpoint and STOP_FOR_COST; do not
continue automatically. Prior observed Notebook entry was3.34h; this is context,
not a promised completion time. CPU/queue/GPU-busy costs remain unknown until
observed. Desktop reconciles when online; no monitor or server takeover.
