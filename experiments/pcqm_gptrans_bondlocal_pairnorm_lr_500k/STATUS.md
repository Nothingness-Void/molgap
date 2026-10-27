# Status

## Terminal acceptance: 2026-10-27 JST

Kaggle1 kernel `141037251` / version `1` completed. Both arms completed
60 passes and strict native and replay checks with no exclusions. The
lower-learning-rate candidate is `NEGATIVE_UNDER_CONTRACT`; the same-run
reference is `CLOSED`. See [qualification](terminal_acceptance/qualification.md),
[decision](terminal_acceptance/decision.md) and [attribution](terminal_acceptance/attribution.md).
Neither model is adopted or released for full training by this decision.

## Submission: 2026-10-27 JST

Kaggle1 `nothingnessvoid/molgap-gptrans-pairnorm-lr-500k`, kernel ID
`141037251`, version `1`, is bound to source commit `a95883ab2a9e28ac08f7832bc428be43e826ae37`
and package `9479db86fadf889d0706421d701e686400042fa49e476bcce8ac28c71fdf8cdd`.
The adapter observed `RUNNING` at 2026-10-26 19:04:05 UTC; this is not evidence
that native all-arm qualification or training has completed or even begun.

Exact launch/response/source observations are retained under `submission/`.
The original local journal and immutable prepared package are under
`D:/molgap-artifacts/runs/pcqm-gptrans-pairnorm-lr-500k-s42/v1/`.

At 19:01 UTC the initial submission invocation returned HTTP403 at
`dataset_status` after journal phase `SOURCE_PUBLISHED`; no kernel was posted.
A read-only account check observed the exact private source dataset ready at
approximately 19:03 UTC. The same prepared package and original journal were
reentered; no new dataset or duplicate kernel was created. The 403 cause is
unknown.

Both models require independent native/terminal acceptance, role and measured
cost evidence and strict replay closure. The historical BondLocal 500K score
is contextual only; the new paired comparison tests learning rate. No model
adoption or full release follows from RUNNING.
