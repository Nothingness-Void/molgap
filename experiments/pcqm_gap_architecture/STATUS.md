# PCQM Gap Architecture Screen Status

- No molecular-research-server access is authorized during architecture
  selection.
- No official PCQM4Mv2 validation or test-dev row is authorized during the
  Kaggle screen.
- Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-ring-hierarchy-cache-s42`,
  version 1, reached `COMPLETE`. Its 100,000/10,000 role cache, 22 shards,
  zero failures, ring memberships and directed relation counts passed local
  no-model acceptance. The accepted aggregate SHA-256 is
  `3f8b271571b8d1026e96fc1dae51d9479489ddd13b73df95740288e6f630779f`.
  The ring GPU successor remains unsubmitted and deferred.
- The active seed-42 question is now the matched local/global allocation
  screen. One sequential Kaggle1 task trains a fresh full-GPS control, a
  block-3/6/9 sparse-attention challenger, and a no-attention shared-GraphState
  challenger under the same geometry, split, initialization seed and training
  contract. Kaggle1 accepted version 1 as
  `nothingnessvoid/molgap-pcqm-local-global-allocation-s42`. Version 1 reached
  terminal `ERROR` after 104.50 seconds, before any candidate or epoch, because
  the subclass reused the parent `MODES` attribute name and caused the valid
  `distance_angle` geometry mode to be rejected. This is an implementation-only
  failure; no scientific result exists. Its evidence is frozen in
  `results/local_global_allocation_seed42/v1_failure_diagnosis.md`. No seed
  43/44 or scientific successor is authorized. The implementation-only repair
  is commit `c61e147796ee4195b837bd7e5639ab0dfe97b12c`; all six static contract
  checks passed. The repaired source dataset reached `ready`, Kaggle1 accepted
  kernel version 2 once, and version 2 was `RUNNING` at the 2026-09-02 14:56
  JST launch check under the unchanged scientific contract. The user then
  selected T4x2 candidate-level parallelism for multi-candidate screens.
  Kaggle's public CLI does not expose cancellation by kernel slug; version 3
  was therefore pushed to the same kernel as the new latest run, explicitly
  requesting `NvidiaTeslaT4` and a 39,600-second timeout. Version 3 was
  `RUNNING` at the 2026-09-02 15:08 JST launch check. It assigns fresh full-GPS
  to physical GPU 0 and sparse-GPS followed by GraphState to physical GPU 1,
  with process-isolated CUDA visibility, RNG, optimizer and checkpoints.
  Version 3 reached terminal `ERROR` after 114.28 seconds and zero epochs.
  Host verification correctly found two Tesla T4 GPUs and CPU initialization
  acceptance passed for all candidates, but CUDA initialization is unsupported
  inside the inherited `fork` workers. The implementation-only repair replaces
  `fork` with two fresh Python subprocesses whose CUDA visibility is fixed
  before import. The retained diagnosis is
  `results/local_global_allocation_seed42/v3_failure_diagnosis.md`. Kaggle1
  accepted the unchanged-contract repair as version 4 with an explicit
  `NvidiaTeslaT4` request. Version 4 reached `COMPLETE`; downloaded artifacts
  passed no-model acceptance. GraphState strictly beat the fresh full-GPS and
  sparse-GPS controls with fewer parameters and higher throughput. The exact
  result is `results/local_global_allocation_seed42/decision.md`. The paired
  seed-43 confirmation is now the only authorized GPU successor; seed 44 is
  conditional on seed 43 passing. No full-data, official-role, desktop, or
  molecular-research-server action is authorized.
- The paired confirmation was submitted once as Kaggle1 kernel
  `nothingnessvoid/molgap-pcqm-graphstate-confirmation-s43`, version 1. Kaggle
  normalized the requested metadata slug to the title-derived identity above.
  The task was `RUNNING` at the launch check on 2026-09-02. GPU 0 trains the
  fresh full-GPS control and GPU 1 trains GraphState under seed 43; no seed-44
  task was submitted.
- Version 1 reached `ERROR` after 0.027 seconds and before preflight because the
  legacy submission client omitted the unsupported `machineShape` field and
  Kaggle assigned one P100. This is an infrastructure-only failure with no
  scientific result. Its diagnosis is
  `results/local_global_allocation_multiseed/seed43_v1_failure_diagnosis.md`.
  An unchanged-contract version-2 retry is authorized through Kaggle CLI 2.2.4,
  which preserves the explicit `NvidiaTeslaT4` request.
- Version 2 was submitted once with Kaggle CLI 2.2.4 and remained `RUNNING`
  after a delayed health check beyond the version-1 host-guard failure point.
  Its launch identity is
  `results/local_global_allocation_multiseed/seed43_v2_launch_manifest.json`.
  No seed-44 task was submitted.
- Version 2 later reached `COMPLETE` and passed no-model acceptance. GraphState
  strictly improved the fresh seed-43 full-GPS control with fewer parameters
  and higher throughput. The exact decision is
  `results/local_global_allocation_multiseed/seed43_decision.md`. This passes
  the conditional gate and authorizes exactly one paired seed-44 T4x2 task;
  full-data, official-role, desktop, and molecular-research-server work remain
  unauthorized.
- The paired seed-44 task was submitted once as Kaggle1 kernel
  `nothingnessvoid/molgap-pcqm-graphstate-confirmation-s44`, version 1, using
  Kaggle CLI 2.2.4 with explicit `NvidiaTeslaT4`. It remained `RUNNING` after a
  delayed health check. Its immutable identity is
  `results/local_global_allocation_multiseed/seed44_launch_manifest.json`.
- Version 1 reached `COMPLETE` and passed frozen no-model acceptance. GraphState
  strictly improved the fresh seed-44 full-GPS control. It therefore improved
  all three paired seeds and their arithmetic mean, so the three-seed gate
  passes and server-side architecture discovery stops at the desktop handoff
  boundary. Exact evidence and the authorized A100 timing-only successor are
  in `results/local_global_allocation_multiseed/decision.md` and
  `results/local_global_allocation_multiseed/desktop_handoff.md`.
- The user authorized continued server-side exploration from the frozen
  GraphState combination. The next single question is Ring-GraphState seed 42:
  a fresh GraphState9 baseline versus the same model plus the already accepted
  deterministic smallest-ring hierarchy. No GPU task is recorded until its
  protocol, implementation, static checks, source package, and launch identity
  are frozen.
- The Ring-GraphState implementation and no-model acceptance are pinned by
  source commit `b9f8445ac400315d82441db8292ea99e68b37dfa`; 14 static contract
  checks passed. Private source dataset
  `nothingnessvoid/molgap-pcqm-ring-graphstate-source` reached `ready`.
  Kaggle1 accepted exactly one T4x2 task as
  `nothingnessvoid/molgap-pcqm-ring-graphstate-s42`, version 1, and it was
  `RUNNING` after a delayed health check. Its immutable launch identity is
  `results/ring_graphstate_seed42/launch_manifest.json`. No confirmation seed,
  full-data, official-role, desktop, or molecular-research-server successor
  was submitted.
- Ring-GraphState version 1 reached `COMPLETE`. Its paired seed-42 gain was
  `0.00020264089107513428 eV`, but throughput fell to 0.8492 times the fresh
  GraphState baseline and both candidate metrics omitted the required input
  cache lineage field. The route closed without confirmation; exact evidence
  is `results/ring_graphstate_seed42/decision.md`. The reusable runner now
  emits the missing field for future jobs, but no Ring-GraphState retry was
  submitted. The only successor is a CPU evidence gate for non-covalent
  ContactState; no GPU successor is authorized before cache acceptance.
- ContactState CPU cache version 1 reached `ERROR` after Kaggle2 rejected the
  inaccessible Kaggle1 private geometry-dataset input. No shard or scientific
  result was produced. The accepted local cache was uploaded unchanged under
  Kaggle2 with the same aggregate SHA, and the unchanged-contract version 2
  was `RUNNING` at the 2026-09-03 04:10 JST health check. The diagnosis and
  launch identity are in `results/contact_state_seed42/`.
- Version 2 reached `ERROR` before reading a shard because Kaggle expanded the
  source archive to `molgap/pcqm_contact.py` while the locator required a
  `src/` prefix. The retained diagnosis is
  `results/contact_state_seed42/cpu_v2_failure_diagnosis.md`. An
  implementation-only path-locator repair is authorized under the unchanged
  CPU contract; no GPU successor is authorized.
- The repaired CPU cache was submitted once as version 3 and was `RUNNING` at
  the 2026-09-03 04:43 JST health check. Its immutable launch identity is
  `results/contact_state_seed42/cpu_v3_launch_manifest.json`.
- Version 3 reached `ERROR` before conversion because the minimal source
  package omitted the `WedgeData` class required to deserialize the accepted
  parent cache. Source-dataset version 2 adds only that frozen class module;
  version 4 was `RUNNING` at the 2026-09-03 05:16 JST health check. Evidence
  and launch identity are in `results/contact_state_seed42/`.
- Version 4 reached `COMPLETE` and passed no-model acceptance: 100,000/10,000
  graphs, zero failures, 3,658,038/366,116 directed contacts, maximum 122/110
  contacts per graph, and aggregate SHA-256
  `49725b92c2c0d33e17633abf8ffa7148ebc8bc9721d3e5b3635f1309891bc826`.
  Its paired seed-42 T4x2 screen then completed and passed no-model acceptance,
  but ContactState was less accurate and slower than the fresh control. The
  route is closed by `results/contact_state_seed42/decision.md`; no extra seed
  or full-data action is authorized.
- Kaggle2 CPU kernel
  `kaseichou/molgap-official-pcqm-gap100k-r1-prep`, version 1, completed with
  a retained acceptance failure because RDKit was absent; its output and log
  remain under `platforms/_records/kaggle/training/pcqm_gap100k_r1_prep_v1`.
- Version 2 included the infrastructure-only fix `rdkit==2025.3.5`, but
  stopped after 65,000 training graphs when a valid bondless molecule
  reached an empty edge-feature range reduction. Its output and log remain
  under `platforms/_records/kaggle/training/pcqm_gap100k_r1_prep_v2`.
- Source commit `e8b190320af98834c9f85f5911d038ac196d1b5b` preserves bondless
  molecules, skips only their empty bond-range reduction, and records their
  count. Version 3 completed but retained one official-train graph failure at
  row `3,003,839`, leaving 99,999 train graphs; strict local acceptance failed,
  its evidence was downloaded, and no GPU task was submitted.
- Source commit `ba82461c53243d733474c8930ac1b86d82451c91` adds a disjoint,
  deterministic 1,024-row reserve stream. It preserves the original 100K/10K
  split hashes, retains every graph failure, and emits hashed failure and
  replacement ledgers. Version 4 is the authorized CPU retry.
- Version 4 passed local no-inference acceptance with exactly 100,000 effective
  train graphs, 10,000 effective internal-validation graphs, one audited
  replacement, and zero unresolved slots.
- Kaggle2 GPU kernel `kaseichou/molgap-pcqm-gap100k-r1-seed42` completed on a
  P100. Its downloaded outputs passed no-inference acceptance, and the exact
  comparison is frozen in
  `results/seed42_structural_vs_edge_state/decision.md`.
- The accepted winner is only the comparator for a materially new pure-2D
  architecture. Seeds 43/44 and full-data training remain unauthorized.
- Learned-query pooling completed, passed no-inference acceptance, failed its
  strict scientific gate, and is closed by
  The complete decision is retained on `archive` and indexed by
  `../_closed/pcqm_server_archive_index.md`.
- Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-local-operators-seed42`, version
  1, completed. Its three required candidates passed terminal no-inference
  acceptance, all failed the strict EdgeState advancement gate, and the
  time-gated fourth candidate was not launched. The exact disposition is in
  The complete decision is retained on `archive` and indexed by
  `../_closed/pcqm_server_archive_index.md`.
 - Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-recurrent-state-seed42`,
  version 1, completed and passed no-inference acceptance, but the recurrent
  graph-state candidate missed the strict EdgeState gate by
  `0.000630125069618209 eV`. Its downloaded evidence remains under the
  platform records; the closure decision is retained on the acceptance branch.
- The sparse topology-wedge CPU retry completed as Kaggle2 kernel
  `kaseichou/molgap-pcqm-gap100k-sparse-triangle-wedge-cache-r2`; its complete
  output and log are retained under
  `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_wedge_cache_r2`.
  No-inference acceptance passed at 2026-08-28 17:00 JST with source commit
  `35fadc9de63e22de7a1cfbe21e4f1af8888e075f`, parent aggregate SHA
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`, cache
  aggregate SHA `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`,
  100000/10000 graphs, 3860510/386070 wedges, 22 hashed shards, and all
  sealed-role flags false. GPU seed-42 submission is authorized after the
  final no-other-GPU check; seed-43/44, full-data, official validation/test-dev,
  and molecular-research-server work remain unauthorized.
- Kaggle2 accepted the corrected GPU submission as kernel
  `kaseichou/molgap-pcqm-triangle-edge-state-r2-s42`, version 1, but the job
  terminated during preflight before the first epoch. Its separate failure
  record is `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_r2_seed42`.
  The fatal error was an `AtomEncoder`/`out_features` API mismatch in the
  sparse wrapper; no training metrics, checkpoint, or candidate result exists.
  The preserved evidence and SHA-256 values are in
  `results/sparse_triangle_edge_state_r2_seed42/failure_diagnosis.md`. No
  retry or successor submission is authorized by this screen.
- On 2026-08-29 the user authorized one implementation-only repair because R2
  never reached an epoch. Source commit
  `76dd6efa76c8236ce80a82a8a43d9f5df426165e` replaces the invalid OGB
  `AtomEncoder.out_features` lookup with the frozen scalar-head input width;
  no architecture, data, seed, precision, batch, optimizer, schedule, cache,
  comparator, or sealed-role setting changed. The source dataset passed its
  ready check and Kaggle2 accepted
  `kaseichou/molgap-pcqm-triangle-edge-state-r3-s42`, version 1. It was
  `RUNNING` at the 2026-08-29 00:10 JST launch check. No cron monitor, second
  retry, seed-43/44 run, full-data run, or official evaluation is authorized.
- R3 later reached terminal `COMPLETE`. Its downloaded outputs passed the
  no-inference acceptance with 4,878,257 parameters, best epoch 38, and a
  validation Gap MAE strictly below the frozen seed-42 comparator. The exact
  result and throughput caveat are frozen in
  `results/sparse_triangle_edge_state_r3_seed42/decision.md`.
- On 2026-08-29 the user authorized the next server-side architecture step:
  one sequential Kaggle2 confirmation task. It must train fresh paired
  EdgeState and Sparse Triangle models at seeds 43 and 44 under
  `sparse_triangle_edge_state_multiseed_protocol.md`. It does not authorize a
  second retry, full-data training, official validation/test-dev, or server
  access.
- The confirmation was submitted once as Kaggle2 kernel
  `kaseichou/molgap-pcqm-triangle-r3-confirm-s43-s44`, version 1, and was
  `RUNNING` at the launch check. The submitted title normalized the requested
  metadata slug by inserting the separator between `s43` and `s44`; both
  identities and the exact source hashes are retained in
  `results/sparse_triangle_edge_state_multiseed/launch_manifest.json`.
- A user-provided Kaggle2 quota screenshot was transcribed at 2026-08-29
  16:05 JST: 29h31m available of 30h, 0h28m used, 0h0m reserved, and 160h55m
  until the displayed reset. The approximate projected reset is 2026-09-05
  09:00 JST; the minute-rounded evidence is in
  `results/sparse_triangle_edge_state_multiseed/quota_snapshot.json`.
- The paired seed-43/44 task reached `COMPLETE`. Its downloaded artifacts and
  hashes passed no-inference acceptance; Sparse Triangle improved EdgeState at
  seeds 42, 43, and 44 and on their mean. The frozen disposition is
  `results/sparse_triangle_edge_state_multiseed/decision.md`.
- The subsequent seed-42 geometry question is frozen in
  `geometry_bottom_fusion_seed42_protocol.md`: distance-only, angle-only, and
  distance-plus-angle features are injected inside every Sparse Triangle
  block. ETKDG cache construction is CPU-only and must pass acceptance before
  the single three-candidate GPU task. No SchNet, late fusion, residual target,
  extra seed, official role, full-data run, or molecular-research-server access
  is authorized by this screen.
- Source dataset `kaseichou/molgap-pcqm-geometry-fusion-source` was created from
  model source commit `e083bee19ee6a13cd9f72e91229752a9d5f56389`.
  Kaggle2 CPU kernel `kaseichou/molgap-pcqm-geometry-cache-s42`, version 1, was
  submitted once and was `RUNNING` at the 2026-08-30 02:48 JST launch check.
  Its immutable launch identity is in
  `results/geometry_bottom_fusion_seed42/cpu_cache_launch_manifest.json`. The
  GPU successor remains unsubmitted until downloaded cache acceptance passes.
- CPU version 1 later reached terminal `ERROR` about 22 seconds after launch.
  It failed before ETKDG because the runner required parent wedge-cache
  `source_commit` `76dd6efa76c8236ce80a82a8a43d9f5df426165e`, while the accepted
  cache manifest correctly records
  `35fadc9de63e22de7a1cfbe21e4f1af8888e075f`. No geometry shard was
  written and no GPU successor was submitted. The preserved diagnosis is
  `results/geometry_bottom_fusion_seed42/cpu_cache_v1_failure_diagnosis.md`.
- The user authorized automatic infrastructure-only diagnosis, repair, and
  resubmission without per-error confirmation. Commit `9132a88` changes only
  the expected parent source identity and adds the missing static assertion;
  all 28 contract tests pass. CPU version 2 completed; the downloaded 110,000-
  role cache, invalid-geometry ledger, 22 shards, and aggregate hash passed
  no-model acceptance. Its identity and acceptance are in
  `results/geometry_bottom_fusion_seed42/cpu_cache_v2_launch_manifest.json` and
  `results/geometry_bottom_fusion_seed42/acceptance.json`.
- GPU version 1 failed before its first candidate because the PyTorch CUDA 12.4
  build omitted P100 `sm_60`. The automatic infrastructure-only repair pinned
  the compatible CUDA 12.6 build without changing the scientific contract;
  dedicated static tests passed. Version 2 then completed all three candidates
  and passed no-inference artifact acceptance. Distance plus angle was the
  seed-42 winner; the terminal disposition is
  `results/geometry_bottom_fusion_seed42/decision.md`.
- No seed-43/44 confirmation, full-data training, official validation/test-dev,
  desktop submission, or molecular-research-server task was submitted by the
  seed-42 geometry screen.
- The separately contracted paired confirmation was submitted once as Kaggle2
  kernel `kaseichou/molgap-pcqm-geometry-confirm-s43-s44`, version 1. Kaggle
  normalized the requested metadata slug by inserting the separator between
  `s43` and `s44`; the runner and scientific contract were unchanged. The task
  reached `COMPLETE`. Its downloaded immutable output passed no-inference
  acceptance: distance-plus-angle improved at seeds 42, 43, and 44 and on the
  mean, while seed 44 was marginal. Exact package, cache, seed, metric, and
  hash identities are in
  `results/geometry_bottom_fusion_multiseed/launch_manifest.json`,
  `results/geometry_bottom_fusion_multiseed/summary.json`, and
  `results/geometry_bottom_fusion_multiseed/decision.md`.
- The separate Kaggle1 CPU torsion derivation completed and passed no-model
  acceptance. The paired GPU comparison was recovered from the version-3
  atomic checkpoint after two documented resume-only infrastructure repairs.
  Kaggle1 version 5 completed both 40-epoch traces and passed independent
  no-inference acceptance with the official validation and test-dev roles
  sealed. Sparse torsion failed the strict seed-42 gate and is closed; the
  accepted distance-plus-angle candidate remains the frozen 100K comparator.
  The final decision and compact hashes are in
  the complete decision and summary are retained on `archive` and
  indexed by `../_closed/pcqm_server_archive_index.md`.
- The post-dual-stream attribution proved that the failed bond stream learned
  but widened the generalization gap; same-information optimizer, width,
  depth, and seed retries are closed. The next distinct mechanism is frozen in
  `ring_hierarchy_seed42_protocol.md`.
- Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-ring-hierarchy-cache-s42`,
  version 1, was submitted once from model source commit
  `58f425258031062c3c3762f13b7d4c160dffba65` and was `QUEUED` at the launch
  check. It derives deterministic smallest-ring nodes and relations from the
  accepted 100K/10K geometry roles. No GPU successor was submitted; cache
  download and no-model acceptance remain mandatory first.

Task order remains authoritative in `ROADMAP.md`.
