# Status — 2026-11-06 JST

Kaggle1 SUBMISSION_BOUND: job143027919/version1; two actual trained arms: current_read, history_read.
Kernel: nothingnessvoid/molgap-k1-history-read-100k-20261106-s42.
Frozen source: e463356bf8e0c110a370e2d2b0aad18de0be0811.
Spec: 654cc8828cc1f85b3327e021b292bbb90873edc578c315db8a4d80c4d999deb6.
Package: 80c27282c74a6fb4ac2cf1771afdc18861ef11d263055d33db6b509f458a9a65.

One T4x2 session;100K/40passes per arm;5h wall/10T4-device-hour ceiling.
Original K1 random initialization plus FP32 Muon; no baseline retraining.
Operations and immutable inputs: D:/molgap-artifacts/runs/pcqm-k1-history-read-100k-20261106-s42/v1/.
Local qualification: verification.md. Native qualification/training evidence remains with the worker.
Luna owns the one-shot scheduler observation. No monitor/cron or shutdown.
On terminal state use strict selective retrieval and existing accept_run/terminal closure for both arms.
Both RML prospective records are published; derived index rebuilt once and check --frozen passed.

Luna observed scheduler running on2026-11-06 09:07:42-45 UTC; scope latest,
observed version1 matches receipt version1; exit0 and zero files downloaded.

2026-11-07 JST: both40-pass arms FINALIZED via strict shared closure. Native qualification/replay VERIFIED, exclusions empty; comparison STRICT_CAUSAL. Current reference CLOSED, history POSITIVE_BELOW_GATE; see decision.md for attribution. New original-Muon calibration is a separate question; no automatic history scale-up.
