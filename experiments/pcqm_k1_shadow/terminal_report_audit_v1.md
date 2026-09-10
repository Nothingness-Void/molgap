# K1 shadow audit v1 terminal report

- Kernel: `kaseichou/molgap-pcqm-k1-shadow-audit`, version 1
- Terminal state: `ERROR`
- Failure stage: label-sealed cache loading, before either model was created or
  executed and before any target-label access
- Root cause: the audit used `hasattr(graph, "y")` on a PyG `Data` object.
  PyG returns `None` for a missing store key, so `hasattr` incorrectly reported
  that every accepted unlabeled graph contained `y`.
- Cache inspection confirmed keys are only `x`, `edge_index`, `edge_attr`,
  `random_walk_pe`, and `row_index`; `"y" in graph` is false.
- Classification: infrastructure/implementation failure, not a scientific
  result and not evidence of label leakage
- Repair boundary: replace only the invalid membership predicate with
  `"y" in graph`; architecture, checkpoints, cache, shadow rows, batch,
  thresholds, bootstrap, and role seals remain unchanged.

The downloaded terminal log is retained in ignored platform record storage at
`platforms/_records/kaggle/training/pcqm_k1_shadow_audit_v1/`.
