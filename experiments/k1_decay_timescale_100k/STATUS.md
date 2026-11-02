# Execution status

On 2026-11-02 Kaggle2 `kaseichou/molgap-k1-decay-timescale-100k-s42` kernel142216564 version1 (original run `k1-decay-timescale-100k-s42-v1` attempt `v1`; journal `E:/MolGap/artifacts/runs/k1-decay-timescale-100k-s42-v1/v1/operations/submission_state.json`) ended with scheduler status ERROR after about 6398 s wall on T4x2 (about 3.55 allocated T4-hours).

- Both arms passed preflight. `wd1e3` trained 40/40 passes, but its owning output check returned BLOCKED (`Development target identity mismatch`): the training worker did not bind the frozen float32 target identity carried by the launch, so development targets were hashed as float64.
- The pair runtime then killed `reference` (SIGKILL) after pass 38/40. It has no output manifest, so `retrieve_family_outputs.py --all-arms --execute` stops at `remote_manifests`.

No acceptance and no scientific claim; protected roles were not used. Unaccepted raw development observations: `reference` best MAE 0.14147 eV at pass 38; `wd1e3` final 0.14116 eV, best 0.14110 eV at pass 36. Continuing or resubmitting required an approved infrastructure fix and an explicit decision (see below).

## Clean rerun (prepared 2026-11-02)

The first attempt failed for an infrastructure reason, not a scientific one. The training-worker target binding and the pair runtime's sibling handling have since been fixed. Under the user's 2026-11-02 rule (infrastructure-caused failures are retried by default; scientific failures ask first), both arms are rerun unchanged as a new logical run `k1-decay-timescale-100k-s42-r2` attempt `v1` ([inputs](retry_v2/experiment.json)), with new Kaggle2 kernel `kaseichou/molgap-k1-decay-timescale-100k-s42-r2` and source dataset `kaseichou/molgap-k1-decay-timescale-100k-src-r2`. The science, arms, seed 42, preset, initialization and roles are unchanged. The budget is the remainder of the original 12 allocated T4-hours after the failed attempt's 3.5547 (8.4453), with a 15201 s wall cap on T4x2. The first attempt's records stay frozen. Nothing has been submitted for the rerun yet.
