# Frozen PairToken selection diagnostic submission

- Kunshan job: `122739196` on `kshdtest`, one Hygon DCU, 30-minute wall ceiling.
- Source commit: `ff8fb6fd` on `molgap-server`; source ZIP SHA-256:
  `9bd292cf71faa55a15eedc44422471ae0c8d7846cb495c66b84da942c78b3c1b`.
- Frozen PairToken checkpoint, PairToken prediction payload, K1-v4 prediction
  payload, and fixed PCQM-100K manifest were remotely checked against the
  protocol's four SHA-256 identities before submission.
- Role: 50,000 official-train-derived development rows only. No training,
  optimizer update, official validation, shadow, test-dev, or challenge access.
- Output: recoverable 5,000-row chunks and atomic progress under the
  `changfeng2006` account's `k1-pair-selection-audit/output` directory. A
  compact aggregate is expected as `diagnostic.json`; no conclusion is
  inferred from submission alone.
