# Sparse Torsion EdgeState GPS9 Seed-42 GPU Decision

Decision date: 2026-08-31

## Question

Does one sparse persistent torsion state improve the accepted distance-plus-
angle Sparse Triangle EdgeState GPS9 on the frozen official-train-derived
PCQM 100K/10K roles?

## Remote outcome

The single authorized Kaggle1 paired GPU task was submitted as
`nothingnessvoid/molgap-pcqm-sparse-torsion-s42`, version 1, and then retried
once under the unchanged scientific contract as version 2. Both versions
reached `KernelWorkerStatus.ERROR` during the implementation preflight, before
the comparator or candidate began training.

| Version | Completed runs | Failure |
|---:|---:|---|
| 1 | 0 | Torsion injection was not zero-initialized |
| 2 | 0 | Finite baseline check failed: maximum absolute difference `0.0037622452`, versus tolerance `1e-6` |

No epoch, checkpoint, validation metric, or model-comparison result exists.
The failure payloads and logs are retained in the two durable platform record
roots listed in [`gpu_failure_summary.json`](gpu_failure_summary.json).

## Disposition

The authorized GPU screen is terminally implementation-blocked. This is not a
scientific non-improvement: the torsion question was never evaluated. The
single infrastructure/implementation retry authority is exhausted, so no
third GPU submission, extra seed, full-data run, official-role evaluation,
desktop submission, or molecular-research-server work is triggered. The
accepted distance-plus-angle model remains the frozen comparator for any
separately authorized future question.

## Evidence

- Launch and terminal identity: [`gpu_launch_manifest.json`](gpu_launch_manifest.json)
- Compact failure hashes and status: [`gpu_failure_summary.json`](gpu_failure_summary.json)
- Accepted CPU cache: [`cache_decision.md`](cache_decision.md)
