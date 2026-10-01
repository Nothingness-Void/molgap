# Kaggle3 dual-arm release

On 2026-10-01, the owning Kaggle adapter rechecked the shared release report and
submitted one private T4x2 screen. The returned job was
`nvoid912/molgap-gptrans-g1-path-and-ema-dual-s42`, kernel ID `136627861`,
version `1`; Kaggle changed the requested slug. The receipt, not the requested
name, owns physical identity: [submission_v1.json](submission_v1.json).

- Arm A: accepted G1 degree initialization plus accepted chemical path mean.
- Arm B: unchanged G1 live training; EMA decay 0.999 instead of 0.9999.
- Frozen G1 comparator reused; no baseline retraining.
- Private source dataset: `nvoid912/molgap-gptrans-g1-path-ema-source`.
- Fixed data: `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`; manifest SHA verified against
  the cross-platform contract.
- Path cache: `nvoid912/molgap-gptrans-author-path-cache-v1`; unchanged accepted
  bytes copied to Kaggle3, without graph rebuilding.
- Local preparation via `prepare-release`: 40.68 seconds after prerequisites;
  this excludes implementation, tests, cache publication and upload. This
  submission therefore does not demonstrate five-minute end-to-end preparation.
- Focused metadata/factory/regression tests: 132 passed. No local training or
  model inference, no protected-role access, no IMS/SCNet or desktop custody.

Both plans were published before submission. Runtime observations, predictions,
checkpoints, costs and terminal evidence still require actual output acceptance.
EMA comparisons use an explicitly qualified intervention world; the ordinary
matched-EMA grouping gate is not relaxed. Terminal closure must verify both
actual Replay pool admissions before reporting Replay-Ready.

The initial exact-identity status check returned RUNNING. The existing Luna B
heartbeat was rebound to the [compact binding](../monitor_binding.json), without
creating a new chat or automation or overriding controller A's runtime. This
authorization did not release successors, extra seeds or larger-scale training.

Local retained release package:
`platforms/_records/kaggle/packages/gptrans_g1_input_ema_v1/release/`.
