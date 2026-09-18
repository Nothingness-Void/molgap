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

The first training allocations reached their 16-hour wall limit with atomic
checkpoints intact; this was a scheduler timeout, not a scientific or code
failure. Reference job `122432964` saved `next_epoch=37` with checkpoint
SHA-256 `061db203ee6a7690399133d1c5cea9d5b579795037fc0bc2ab324e2faacb3735`.
Pair PreNorm job `122432972` saved `next_epoch=39` with checkpoint SHA-256
`3c364e10cfb2a9c33726bcc9eb4a1faa0972a2b779f5d51faabf97eebaca6465`.

One unchanged-contract continuation per arm was submitted through the
versioned atomic-resume wrapper at commit
`2e610044eb376c95f46bb5df9f7b073ed51ac7dd`, archive SHA-256
`dc35a31a43238c13a7a626be9d44f23651fd428568313316600967137212827f`:

| Arm | Continuation | Resume cursor | Initial state |
|---|---:|---:|---|
| reference | `122518428` | epoch 37 | pending for priority |
| Pair PreNorm | `122518430` | epoch 39 | pending for priority |

The scientific source, cache, model, optimizer, schedule, precision and batch
remain the frozen retry identities above. The wrapper only supplies the
already-supported `--resume` path and writes to a new attempt directory.

Reference continuation `122518428` failed before checkpoint restore because it
reran the stochastic calibration and observed a one-ULP parameter delta above
the unchanged `1e-7` gate. Pair job `122518430` was still unallocated and was
cancelled with zero runtime to avoid the same redundant gate. The diagnosis is
`results/resume_infrastructure_failure_122518428.md`.

The certified-resume repair reuses the original accepted runtime certificate
only after verifying its ID, the current runtime fingerprint, frozen scientific
source/archive and checkpoint contract. It does not loosen calibration. New
execution source commit `7d7b3b046f2048bc841226d14acedbbfd194c422`, archive
SHA-256 `a1d30b15363d85260d36a41503d38394b22ef949f81f56177aad106c0e3d615a`:

| Arm | Certified continuation | Resume cursor | Initial state |
|---|---:|---:|---|
| reference | `122520066` | epoch 37 | pending for priority |
| Pair PreNorm | `122520074` | epoch 39 | pending for priority |
