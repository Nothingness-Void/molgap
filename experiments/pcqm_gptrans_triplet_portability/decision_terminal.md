# Frozen triplet transfer interpretation — 2026-10-10

## Decision

The NO_TRAIN run completed and passed independent saved-output acceptance.
The predeclared scale-consideration gate did not pass: the later-cohort point
gain retained more than50% of the100K gain, but its paired95% interval included
zero. No500K training, extra seed, checkpoint reselection or automatic successor
was released. This is uncertain positive frozen-cohort evidence, not a negative
100K architecture result and not proof that triplet communication is useless.

Authority: [protocol](protocol.md); [independent acceptance](results/acceptance_summary.json),
[retrieval pins](retrieval_artifacts_v1.json), and [physical receipt](submission_receipt.json).

## Endpoints

| Role | Rows | Parent MAE(eV) | Aggregation MAE(eV) | Paired gain(eV) |
|---|---:|---:|---:|---:|
| Original100K internal development | 50000 |0.142358227481246|0.139308451203704|0.003049776277542|
| Fixed later500K internal-development cohort |10000|0.147057610523701|0.145290646684170|0.001766963839531|

The later point gain retained57.937469% of the accepted original100K gain.
Candidate-minus-parent95% row-bootstrap interval was
[-0.003781002392769,+0.000165530472398]eV; candidate won50.05% of molecules.
The frozen point threshold was0.001524888699055eV (half the accepted100K gain),
with the separate requirement that the interval's upper bound be below zero.
Do not relax that requirement after seeing this result or infer seed variation
from the row bootstrap.

The full original50000 predictions reproduced with maximum difference
0.000001907348633eV and MAE difference0.000000001120567eV. This rejects a gross
loader/EMA/transform mismatch as the explanation of the attenuated later gain.

## Attribution and limits

The gain changed without any optimizer step or new training exposure. Therefore
this particular attenuation has a molecular-cohort/sampling component; it
cannot be solely a consequence of insufficient500K training exposure. The
diagnostic does not measure500K optimization, long-run convergence or full-scale
ranking, and cannot quantify their separate effects.

Parent-error-ranked post-hoc groups showed gains on high-parent-error rows and
losses on low-parent-error rows. These groups use target-derived reference
errors and have selection/regression-to-the-mean effects. They do not establish
chemically identifiable specialists, a deployable Router, or a causal mechanism.
The cohort was previously used for research, not a fresh sealed audit.

## Execution and evidence

Remote frozen inference used seed42, FP32/noTF32, deterministic algorithms and
BS128. Accepted parent predictions were reused; neither model was trained.
The model state was unchanged; official validation, test-dev and challenge
remained untouched. All21 manifest-bound output files were retained and hashed.
Local acceptance read saved prediction tensors only; no local model inference.

The observed Python allocation window was109.176768083seconds, with two T4s
allocated and one worker. Both devices were charged in the native ledger:
218.353536166allocated-device-seconds, or0.060653760T4-hours. Queue and provisioning
before Python were unavailable; this is not a claim of exact Kaggle billing.

RML classification is NO_TRAIN / PAIRED_ENDPOINT, strict_ready=false and
training_replay_ready=false. It closes a complete applicable diagnostic chain,
not an additional canonical training Replay pair. The accepted100K aggregation
and attention training Replay pairs remain unchanged.
