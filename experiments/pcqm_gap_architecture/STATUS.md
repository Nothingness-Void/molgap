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
- No GPU task, seed-43/44 run, full-data run, or molecular-research-server work
  is active. The next authorized task is one seed-42 recurrent graph-state
  EdgeState screen under `recurrent_graph_state_seed42_protocol.md`; its source
  and remote kernel must pass static and manifest checks before submission.

Task order remains authoritative in `ROADMAP.md`.
