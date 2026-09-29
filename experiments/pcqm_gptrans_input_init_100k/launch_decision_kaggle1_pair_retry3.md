# Kaggle1 paired input-initialization retry declaration — 2026-09-29

The first submitted T4x2 attempt stopped during candidate GPU preflight before
training because the candidate state hash produced by Kaggle's Torch runtime
did not match the locally frozen Torch hash. The reference preflight passed.
The retained startup log and `decision_kaggle1_pair_v1_infra.md` own that
observation. The intervening custom-launcher retry plan was never published to
Kaggle and closed `NO_TRAIN` in `decision_kaggle1_pair_retry2_no_train.md`.

Authorize one source-frozen paired version on Kaggle1 using the repository's
existing GPTrans T4x2 runner. Its reference and input-initialization arms run
concurrently on separate T4s with the same accepted graphs, seed, V4 training
recipe and development role. Pin the candidate complete state to the observed
Kaggle Torch hash in `training_contract_kaggle1_pair.json` and the new Spec.
Both GPU preflights must pass before training. Apply the frozen paired 100K
decision gate; no reference promotion follows from a queue state or one score.

A repeated initialization mismatch, failed preflight, or missing source/data
identity stops this retry without another automatic submission. The actual
Kaggle version, source/package hashes, per-arm native cost and terminal
artifacts must be reconciled before any scientific decision.
