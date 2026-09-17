# Status

Frozen source commit: `8428c7b9832e1ad60ce44522184221652d98bd12`.
Source archive SHA-256:
`cc1b72fb360e485c504fafbbc2e4f6d3999be6cebac68c5b0f4d0c8aad03c202`.

Kunshan jobs submitted on 2026-09-18:

| Arm | Preflight | Training | Release rule |
|---|---:|---:|---|
| reference | `122425237` | `122425440` | training has `afterok:122425237` |
| Pair PreNorm | `122425241` | `122425441` | training has `afterok:122425241` |

The initial preflights failed before training because the wrapper incorrectly
required bitwise-identical full-model optimizer states. Their dependent jobs
never allocated. Diagnosis:
`results/infrastructure_failure_2026-09-18.md`.

The unchanged-contract infrastructure retry uses source commit
`dff104723222346e703cd98ef71c19884f46076c` and archive SHA-256
`6f0de8e074a2b07da145353904062b656e319724819543bd986ccc35ad4b02f3`:

| Arm | Retry preflight | Retry training | Release rule |
|---|---:|---:|---|
| reference | `122432949` | `122432964` | training has `afterok:122432949` |
| Pair PreNorm | `122432955` | `122432972` | training has `afterok:122432955` |

Low-cost monitor B heartbeat:
`molgap-gptrans-v5-kunshan-retry-monitor`. Healthy queue/training states are
silent; terminal success or actionable failure is handed once to server A.

No scientific result is assumed until both arms pass independent artifact
acceptance and the paired comparison is recomputed without model inference.
