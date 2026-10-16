# Unified PCQM roles V2 evidence

This directory retains the 2026-10-16 CPU implementation/preparation evidence.
Read [live state](../../../../CURRENT_STATE.md) first. The
[acceptance decision](../../../../docs/operations/CROSS_SCALE_V1_HARDENING_ACCEPTANCE.md)
owns interpretation; [final machine closure](closure.json) pins its
[detailed base evidence](acceptance.json), exact source, test groups and raw
artifacts. This is not an executable training package.

- [Frozen inventory](role_inventory.json), [original rule](rule.json),
  [repaired source-bound rule](rule-r2.json), [CPU launch receipt](cpu_submission.json).
- [Raw failure](failure.json), [last durable progress](progress.json).
  A retained RUNNING progress field is a historical cursor, not live status.
- [Counterexamples](counterexamples.json), [RML projection summary](rml_backtest_summary.json),
  [synthetic JSON/Markdown report](report_example/comparison.md).

Ignored storage holds source archives, original `pcqm-unified-v2.o1730791`,
JUnit XML and the full RML projection. Their hashes are retained in compact
records. No model weights, real predictions or Audit labels are stored here.
Dataset role manifests/source-index SHA do not exist because preparation
failed before selection. No private mirror or accepted data export is implied.
