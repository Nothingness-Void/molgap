# Operational status

2026-09-14: Kaggle2 `kaseichou/molgap-gptrans-relation-flow-s42` version 1
was submitted and returned RUNNING. The pulled metadata confirmed T4x2 and
the fixed 100K dataset. The remote entrypoint matched after newline
normalization. GPU model checks/preflight have not yet been accepted locally.

Two independent candidates: `pair_prenorm`, `centered_logits`. Submission
identity is frozen in `results/submission.json`; static no-model tests: 8 passed.
Source publication used the package directory as cwd with `-p .` because the
Windows Kaggle uploader mishandled slash-containing staging paths; these were
source-upload failures, not GPU runs or scientific retries.

One of two newly authorized rounds has been submitted. The second remains
controller-owned, requiring terminal acceptance and failure/success attribution.
No desktop, SCNet, IMS, scale or official-role task is authorized here.

Output destination:
`platforms/_records/kaggle/training/gptrans_relation_flow_s42_v1/`.
Mechanical acceptance command and monitor routing: `monitor.md`.

Heartbeat `molgap-luna-training-monitor` is ACTIVE on the existing monitor B,
every 30 minutes. B was explicitly dispatched with `gpt-5.6-luna` / `max`;
controller A model/reasoning settings are not overridden. Normal RUNNING is
quiet; terminal evidence wakes A for attribution and the remaining round.
