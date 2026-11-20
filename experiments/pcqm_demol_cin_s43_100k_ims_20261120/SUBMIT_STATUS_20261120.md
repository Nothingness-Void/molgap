# IMS submission status, 2026-11-20 night

> **Superseded 2026-11-20 (user-reported; scheduler not re-queried by this update).** Jobs 2077090, 2077091, 2077093 and 2077094 below were cancelled by the user. They are replaced by a single serial IMS job that has not been submitted yet. The J1/J2 specs in `protocol.md` are superseded by that job. No result exists for these job ids.

Submitted by the Sonnet executor after Opus pre-submit review (no blockers). Not pushed.

| Job | Run id | Preflight | Training | Walltime | --wall-s |
|---|---|---|---|---|---|
| J1 DeMol-CIN 100K s43 | pcqm-demol-cin-100k-ims-20261120-s43 | 2077090 | 2077091 | 6h | 19800 |
| J2 GPT-T 100K s43 | pcqm-gptrans-pun-100k-ims-20261120-s43 | 2077093 | 2077094 | 4h | 12600 |

Observed in `jobinfo -c` right after submit: J1 preflight Queue (gpu); J2 preflight Queue with reason `(other)` (cause unread; do not jdel or resubmit). Training jobs Hold (afterok). Cutlass/GRIT r2 jobs 2066226/27 and 2066230/31 unchanged.
Group ew6 credits before submit: 930,543 remaining; estimate ~841 points.

Opus Monday notes:
- Compute rule (b) from the scorer's per-row predictions, not from attribution contracts (experiment_attribution.py:670 rejects Kaggle s42 dev identity b7d8bc6d vs IMS 3270b300).
- Pass --expect-dev-mae by hand; selected_eval.json is not yet read by the scorer.
- IMS RAM limit unmeasured; OOM would fail closed.
- Every IMS verdict is non-strict (nomination, not adoption).
- 5 failures in tests/test_ims_v4_submit.py pre-exist (identical on b028886c6 and 88a3e64af).
- F0 (local, before submit): tail MAE DeMol 149.389 vs GPT-T 155.491 meV, delta -6.102.
