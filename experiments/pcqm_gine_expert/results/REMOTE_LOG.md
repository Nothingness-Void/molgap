# PCQM GINE expert — completed remote rounds

Dated Kaggle round record for the benchmark-only PCQM Gap specialist. Official
test and the future sealed 20K were never accessed, and the production registry
is unchanged in every round below. Fixed official-valid is a 5K split read only
after a version froze.

## Pilot (rejected at the scale gate)

Kernel `nothingnessvoid/molgap-pcqm-gin-expert-pilot`, version 3. Its 11 graph
shards and all declared artifacts passed count, uniqueness, finite-label,
loadability, and SHA256 checks. Fixed official-valid 5K Gap MAE `0.213504 eV`:
`0.078186 eV` better than routed v4, but above the predeclared `0.20 eV` gate, so
it is rejected as a hierarchical-Oracle prerequisite and authorizes no Router or
GPS9/fusion expansion.
Decision: `decision.md`. Acceptance: `acceptance.json`.

## v4 continuation (accepted as the Oracle prerequisite)

Resumed the accepted epoch-29 optimizer/scheduler/scaler state, reused all 11
validated shards, and selected epoch 48. Fixed official-valid 5K Gap MAE
`0.196598 eV` — `0.095092 eV` better than routed v4, clearing the `0.20 eV` gate
by `0.003402 eV`. Accepted only as the task-level PCQM Gap prerequisite for the
planned Oracle study; no learned Router, GPS9/fusion expansion, sealed-set
access, or registry change is authorized.
Accepted artifacts: `nothingnessvoid/molgap-pcqm-gin-v4-accepted-20260724`.
Decision: `continuation_v4_decision.md`.

## v5 continuation (accepted warm start for the local branch)

Best epoch 68 reaches `0.191690 eV` on the frozen scaffold development split and
`0.187320 eV` on fixed official-valid 5K, improving accepted v4 by `0.009278 eV`.
Prediction MAE was independently reproduced and all downloaded artifact hashes
match. This is a 250K-sample local protocol result, not a leaderboard score. It
remains the accepted warm start for the local v6/v7 branch.
Decision: `continuation_v5_decision.md`.
Accepted checkpoints: `nothingnessvoid/molgap-pcqm-gin-v5-accepted-20260726`.

## GPS9-320 architecture pilot (rejected, configuration invalid)

Best development MAE `0.462255 eV`, training non-finite from epoch 15, fixed
official-valid 5K Gap MAE `0.491629 eV`. This invalidates the
implementation/training configuration, **not** GPS as a model family — it lacked
the published positional-encoding and optimization protocol. Do not resume or
scale this checkpoint.
Decision: `gps9_320_pilot_decision.md`.

## B-based PCQM Gap head pilot (closed)

PCQM4Mv2 official-train scan `703665` produced an accepted 200K raw hard pool. A
domain audit found 103,440 radicals, so the raw pool must not be trained on; the
fixed clean pool has 95,909 rows. Because retention variant B already improved
common/OOD/P8-hard, a separate B-based specialist was tested: `706147`
materialized the clean pool and embeddings, `706148` trained the output head,
`706149` evaluated it. All three completed, but the candidate regressed
common/OOD/P8-hard average by `+0.01615/+0.02153/+0.01065 eV` and PCQM Gap by
`+0.08837 eV`. The frozen-head specialist is rejected and closed.
Configuration: `../gap_head_pilot/experiment_manifest.json`.
Decision: `../gap_head_pilot/decision.md`.
Source decision: `../train_residual_scan/decision.md`.
