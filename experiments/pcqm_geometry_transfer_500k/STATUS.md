# Status

The immutable source bundle from commit
`01d5c88cb4118e9cc6888440a5e1842ddaf12988` was submitted to SCNet Kunshan on
2026-09-12. Its SHA-256 is
`722f32cfc5c20f6de4f1decdf8835c7541c74399daf9422210b4712faf697f4b`.

The bounded dependency chain is:

| Role | Job | Initial state | Dependency |
|---|---:|---|---|
| GPTrans-T geometry preflight | 121849083 | PENDING/Priority | none |
| K1 geometry preflight | 121849089 | PENDING/Priority | none |
| GPTrans-T geometry 500K | 121849091 | PENDING/Dependency | afterok:121849083 |
| K1 geometry 500K | 121849093 | PENDING/Dependency | afterok:121849089 |
| aligned development fusion | 121849105 | PENDING/Dependency | afterok:121849091:121849093 |

The two preflights have 30-minute limits, each training task has a 24-hour
limit, and fusion is CPU-only with a 30-minute limit. Outputs, logs, atomic
checkpoints, and prediction payloads are isolated under the experiment's
durable remote root. No official validation, test-dev, or test-challenge role
was opened, and no production model was changed.
