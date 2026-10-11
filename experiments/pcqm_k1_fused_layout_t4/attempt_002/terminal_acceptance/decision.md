# Attempt002 partial infrastructure failure

Observed2026-10-11 JST. Exact Kaggle1 kernel138119131, submission version1,
`nothingnessvoid/molgap-k1-fused-layout-100k-s42-v2`. The
[scheduler/retrieval observation](../submission/terminal_observation.json) and
[checked partial report](acceptance.json) own this result. The output-list API
addresses the latest session, not an independently pinned version; retained
runtime source, Spec, arm and recipe identities match the frozen launch.

Both native T4 preflight workers exited0. Both formal training workers then
acknowledged one epoch:781 optimizer steps and99968 sample presentations each.
The original/optimized internal-development trace MAEs are0.361787/0.332582eV.
These are first-epoch observations, not accepted40epoch endpoints or evidence
that the candidate satisfies quality noninferiority.

## Attribution

Both workers fail at frozen `k1_screen_training.py:917`:
`atomic_json(timing_path, timing_rows)`. The caller supplies a list of timing
rows, while `training_reproducibility.py:169` converts its Mapping input with
`dict(value)`. Both retained train logs report the same ValueError. A CPU-only
fixture reproduced that interface exception without model construction,
training, inference, CUDA or role access. The earlier initialization-container
fault no longer blocks this attempt.

The exception occurs after the first-epoch selected prediction, checkpoint and
canonical trace writes, but before `execution_times.json` and the complete
family output manifest. Remote checkpoint/prediction paths are present; their
bytes were not downloaded or locally validated in this reconciliation. No
claim of resumable acceptance or complete artifact verification follows.

This is a shared logging-interface failure, not divergence or a negative model
result. Train-only timing is unavailable. Trace epoch windows include train,
development and IO, so their161.80/158.10seconds cannot establish the frozen
10% training-pipeline speed gate. Both speed and quality gates stay unevaluated.

Entry wall437.93seconds corresponds to875.85 allocated T4-device seconds
(0.24329device-hours) within the retained two-device entry window. Queue,
entire platform allocation, billing and GPU-busy cost remain unknown.

## Disposition

Mechanical: verified partial infrastructure failure. Science: INCONCLUSIVE,
incomplete40epoch endpoint. No adoption, scale-up, automatic retry or
continuation. The next discriminator is a repaired timing interface with a
first-epoch output-write integration regression, before any separately
authorized remote work. Formal V5/RML terminal closure remains pending; the
prospective records are preserved, not presented as replay-ready records.
