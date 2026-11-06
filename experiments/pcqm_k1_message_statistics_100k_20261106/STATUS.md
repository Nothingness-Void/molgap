# Status — 2026-11-06 JST

Kaggle1 SUBMISSION_BOUND: job143027803/version1; two actual trained arms: message_mean, message_std.
Kernel: nothingnessvoid/molgap-k1-message-statistics-100k-20261106-s42.
Frozen source: b08877af8614ab12a3b9de392ada28b11d7e4867.
Spec: 5ca882b531ed4e80a66d30442b28181d43e8a370a19b67a9609bbec692c1e1f8.
Package: c2e4c66f9164fff16d7ddc82d80f82e0917cefebfce01bd077a384ce254e471d.

One T4x2 session;100K/40passes per arm;5h wall/10T4-device-hour ceiling.
Original K1 random initialization plus FP32 Muon; no baseline retraining.
Operations and immutable inputs: D:/molgap-artifacts/runs/pcqm-k1-message-statistics-100k-20261106-s42/v1/.
Local qualification: verification.md. Native qualification/training evidence remains with the worker.
Luna owns the one-shot scheduler observation. No monitor/cron or shutdown.
On terminal state use strict selective retrieval and existing accept_run/terminal closure for both arms.
Both RML prospective records are published; derived index rebuilt once and check --frozen passed.

Luna observed scheduler running on2026-11-06 09:07:31-33 UTC; scope latest,
observed version1 matches receipt version1; exit0 and zero files downloaded.

2026-11-07 JST: both40-pass arms FINALIZED via strict shared closure. Native qualification/replay VERIFIED, exclusions empty; comparison STRICT_CAUSAL. Mean reference CLOSED, std POSITIVE_BELOW_GATE; see decision.md for attribution. No automatic dispersion scale-up or successor.
