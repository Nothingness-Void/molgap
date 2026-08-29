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

Task order remains authoritative in `ROADMAP.md`.
