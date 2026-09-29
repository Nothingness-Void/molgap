# GPTrans V5 audit-reference execution status

Observed 2026-09-29 16:21 UTC. The user authorized exactly one matched 100K
baseline, not a candidate or full-scale run. Source archive
`af94a63c3c4626ceec8af4106aad0ea97d398e40aefb04c51b15356b994137a3`
was built from scientific-source commit
`21f70ab93c3b0ddc4745316daf8459df40efdb48`; the private Kaggle2 source
dataset `kaseichou/molgap-gptrans-v5-audit-source-v1` is ready at version 1,
and a downloaded `source_payload.bin` matched the archive SHA. The original
initial-state dataset and accepted fixed 100K graph dataset are ready.

The first physical submission,
[`kaseichou/molgap-gptrans-v5-audit-reference-seed42-segment1`](https://www.kaggle.com/code/kaseichou/molgap-gptrans-v5-audit-reference-seed42-segment1),
had the wrong title-derived slug and assumed Kaggle would publish additional
kernel package files. Remote source pull showed only script and metadata,
and its mounted dataset list omitted the new source. Its frozen SHA/function
gate therefore cannot reach training. It was `QUEUED` at the observation;
keep it as a separate infrastructure attempt, never a scientific result.

The corrected first 10-epoch segment,
[`kaseichou/molgap-gptrans-v5-audit-reference-s42-v1`](https://www.kaggle.com/code/kaseichou/molgap-gptrans-v5-audit-reference-s42-v1),
was submitted as version 1 with requested `NvidiaTeslaT4`. Remote metadata
ID `136421626` confirms the exact kernel slug, T4 machine shape, and three
mounted datasets (V5 source, V4 initial state, fixed 100K graphs). Scheduler
status was `RUNNING` but no startup log or scientific output had appeared at
the observation. No preflight, epoch, or MAE has yet been accepted. The
prospective trajectory is [`rml_plan/trajectory.json`](rml_plan/trajectory.json).

Next reconciliation: inspect the corrected kernel's preflight output and
log; only an accepted preflight plus an independently retrieved and SHA-
validated 10-epoch checkpoint/trace may authorize the next segment. The
failed first submission must be allowed to reach a terminal infrastructure
status and recorded separately. Neither G1 nor G2 is released by queue state.

The local thread heartbeat `gptrans-100k` checks this exact chain every 30
minutes, stays silent while status is unchanged, and may submit the next
segment only after the preceding output passes `accept_segment.py`. It must
stop itself after full acceptance or an actionable block.
