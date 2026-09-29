# Kaggle1 pair retry 2 decision — 2026-09-29

The first Kaggle1 T4x2 attempt stopped during candidate initialization
verification before either arm trained. `decision_kaggle1_pair_v1_infra.md`
owns that inconclusive outcome and the retained remote log. The candidate's
observed complete state hash under that T4 runtime is
`9af1db8997c8ac28dc085ee113fef23c639f700f45d915e40c4339ef2e181c6a`.
The original local hash is inapplicable to that remote Torch build.

Freeze a new two-arm Spec with the observed T4 candidate hash and the same
source mechanism, seed, graph rows, V4 recipe, accepted Kaggle1 control,
preflight and 3 meV paired gate. Retain the same actual private kernel identity
for a single version-2 retry, but use a new private source dataset identity to
prevent the failed declaration from being silently overwritten. Both arms
must pass their GPU preflights before training. A repeated hash mismatch or
any new source/data/runtime failure stops; no further automatic retry or
scale-up is authorized by this decision.
