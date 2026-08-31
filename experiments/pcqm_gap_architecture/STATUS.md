# PCQM Gap Architecture Screen Status

- No molecular-research-server access is authorized during architecture
  selection.
- No official PCQM4Mv2 validation or test-dev row is authorized during the
  Kaggle screen.
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
  `results/query_pool_seed42/decision.md`.
- Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-local-operators-seed42`, version
  1, completed. Its three required candidates passed terminal no-inference
  acceptance, all failed the strict EdgeState advancement gate, and the
  time-gated fourth candidate was not launched. The exact disposition is in
  `results/local_operator_search_seed42/decision.md`.
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
- The accepted distance-plus-angle candidate is the frozen 100K comparator for
  the one seed-42 sparse torsion-state task. Its protocol and source/cache
  identity must be accepted before any GPU submission; no torsion GPU task has
  been submitted.
- The separate CPU torsion derivation is running on Kaggle1 as
  `nothingnessvoid/molgap-pcqm-sparse-torsion-cache-s42`, version 1. It
  completed and passed no-model acceptance; its account1 input mirrors,
  canonical slug, source marker, retrieval contract, and accepted output are
  recorded in
  `results/sparse_torsion_edge_state_seed42/cache_decision.md`.
- The one strict paired seed-42 GPU screen is running on Kaggle1 as
  `nothingnessvoid/molgap-pcqm-sparse-torsion-s42`, version 1, with P100 and a
  23,400-second bound. Its comparator/candidate order, accepted cache hash,
  and no-sealed-role flags are recorded in
  `results/sparse_torsion_edge_state_seed42/gpu_launch_manifest.json`.

Task order remains authoritative in `ROADMAP.md`.
