# K1 pretrained consistency with mean-output teacher

Desktop-owned 100K pair authorized on 2026-10-05. Start from desktop
`3f194a932fe2da3d8ac14ecedffac709d90c3de7`; retain this question on
`codex/exp/k1-pretrained-consistency-teacher` through acceptance and disposition.

Read [protocol](protocol.md), [evidence review](evidence_review.md),
[initialization provenance](initialization_provenance.json), and
[status](STATUS.md). Existing workflow/platform adapters own preparation,
submission and terminal closure. Each arm has its own prospective RML record.

| Arm | New recipe |
|---|---|
| pretrained_consistency | Retained stage-10 K1 backbone + restored original Gap head + two-pass consistency |
| pretrained_consistency_teacher | Identical initialization and recipe + fixed teacher MSE on mean dropout prediction |

No module or model is adopted by preparation or submission. Track A and the
accepted full EdgeState reference retain their own decisions.
