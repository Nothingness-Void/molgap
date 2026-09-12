# Status

The immutable source bundle from commit
`01d5c88cb4118e9cc6888440a5e1842ddaf12988` was submitted to SCNet Kunshan on
2026-09-12. Its SHA-256 is
`722f32cfc5c20f6de4f1decdf8835c7541c74399daf9422210b4712faf697f4b`.

The bounded dependency chain reached terminal state as follows:

| Role | Job | Initial state | Dependency |
|---|---:|---|---|
| GPTrans-T geometry preflight | 121849083 | COMPLETED | none |
| K1 geometry preflight | 121849089 | COMPLETED | none |
| GPTrans-T geometry 500K | 121849091 | COMPLETED | afterok:121849083 |
| K1 geometry 500K | 121849093 | COMPLETED | afterok:121849089 |
| aligned development fusion | 121849105 | FAILED before payload read | afterok:121849091:121849093 |

Both 500K runs retained complete best models, validation payloads, checkpoints,
metrics, and hashes. GPTrans-T reached `0.1027374789 eV`; K1 reached
`0.1043544337 eV` on the aligned development role. The first fusion attempt
could not import DTK PyTorch on the CPU partition because the OpenMPI and HSA
runtime libraries were absent. Retry `121892309` confirmed the second missing
library after OpenMPI was added. Retry `121892470` used one short DCU allocation
to provide the unchanged DTK runtime and completed in 126 seconds without
retraining either encoder. The accepted fusion metrics are recorded in
`results/fusion_metrics.json` and the interpretation is in `decision.md`.

Outputs remain isolated under the experiment's durable remote root. No official
validation, test-dev, or test-challenge role was opened, and no production model
was changed.
