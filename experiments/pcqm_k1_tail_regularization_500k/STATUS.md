# Status

Owner: `codex/exp/k1-tail-regularization-500k`, based on desktop7aa6814b.
Kaggle1 [kernel140639445/version1](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-tail-wd-500k-s42)
is submitted. The source response, exact receipt and byte-verification records
are retained in [submission](submission/launch_receipt.json).

Frozen executable source873c3ac4, Spec
`f5404cb801a792bf57f56dfde0b496e8f21a7ad67b9f22f93a8721cc1d04a994`, package
`09efe4c0b754b0c7b7f2d1ea8e7a679be3b0a7b2763eb30a8afc9fff131c1e32`.
Arms `tail_wd1e3` and `tail_wd1e2` each target20passes/78,120new optimizer steps.

Luna verified actual startup at2026-10-25 04:08JST (2026-10-24 19:08UTC):
both arms completed pass1 with3,906steps/499,968sample presentations and remained
RUNNING without a failure message. Their fail-closed initial predictor checks
therefore passed before formal updates. [Exact startup observation](submission/startup_verified_observation.json)
retains source resolution, native preflight/train phase starts and both TRAIN_LOGs.

| Arm | Pass1 LIVE development MAE/eV | Shadow EMA MAE/eV | Invocation elapsed/s |
|---|---:|---:|---:|
| tail_wd1e3 |0.10771922767162323|0.10591693222522736|665.05886826|
| tail_wd1e2 |0.10838806629180908|0.10583055764436722|666.768229678|

These are first-pass progress points, not final benefit or an early-stop gate.
No terminal acceptance, replay-ready result or model adoption is claimed.

The source create call lost its response with HTTPError. Read-only reconciliation
proved the exact private source version1 ready; the original journal then
verified content and continued to this one bound kernel. No second source or
kernel was created. The diagnostic-observability gap was reported to the
infrastructure owner; the frozen training package remains unchanged.

Durable operation root:
`D:/molgap-artifacts/runs/pcqm-k1-tail-regularization-500k-s42/v1/operations`.
One existing Luna/max conversation owns30minute same-chat heartbeat checks.
Its ID is `01a168cc-d27b-7363-bccf-cbdecc42c4d4`, titled
`Kaggle1 K1 500K 后段双臂监控`; it notifies the parent once at terminal state.
The primary agent does not poll runtime. Desktop offline time is allowed; no
server transfer or takeover. Local heartbeat checks require this desktop online;
the Kaggle job runs independently. The startup condition for the user's shutdown
request is satisfied. The exact shutdown command result is reported separately.
