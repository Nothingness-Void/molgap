# Terminal decision — 2026-10-08

Outcome: **NO_TRAIN**. Accepted execution-only A100 diagnostic; no model
promotion, training replay qualification or successor release.

The missing `molgap.pcqm_wedge.WedgeData` dependency caused attempt001 to stop
before graph loading completed or scratch execution began. Attempt002 included
the exact retained module and used a separate durable directory; the selected
checkpoint,4096 sampled training members, objective and FP32 settings were unchanged.

All four timing cases, six synchronized phase observations, three operator
profile steps and four gradient batches completed. Twelve worker-finalized
artifacts passed exact SHA256 verification. The returned process completed in
63.702s including imports, within1200s. A100 was released and Manage Sessions
was empty; the notebook was closed after the Drive evidence bundle was saved.

This supports prioritizing the two-forward cost over loader-worker tuning in
this bounded A100 condition. Single-forward normalized L1 changes supervision,
dropout and BN updates, so its speed is not evidence of equivalent MAE. Historical
T4 epoch cost, full-data loading/evaluation/publication and trajectory-wide
accuracy causality remain unqualified. Read [attribution](attribution.md).

Scientific continuation requires a separately frozen native T4 execution or
single-pass quality question. The existing Kaggle two-forward consistency pair
is a different discriminator; this decision grants no resubmission or new run.

Git route: accepted diagnostic with reusable implementation; eligible for the
BRANCHES non-promotion desktop route after integration review.
