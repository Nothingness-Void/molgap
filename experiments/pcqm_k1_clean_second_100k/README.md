# K1 clean second training view at100K

Desktop-owned Track B question on `codex/exp/k1-clean-second-100k`, based on
desktop `4f21cde249aea3710359039663d15c83656d9574`.
Proposed run: `molgap-k1-clean-second-100k-s42-v1`.
Family: `k1-clean-second-view`.

Does replacing only the second stochastic training view with a Dropout-off
view improve clean development Gap MAE against a fresh mean2 control?
[Protocol](protocol.md) owns the arms, gate, authorization and resource scope;
[evidence review](evidence_review.md) owns the rationale and evidence limits.

The human user explicitly authorized four fresh arms across two notebooks
within15 allocated T4 device-hours on2026-10-10. This directory owns slot2:
seed42 `mean2` reference versus `clean_second` candidate in the same physical
job. Slot1's seed43 mean2/single question remains separately owned. Historical
mean2 weights or results are not an accepted reference for this fresh pair.

Reuse [shared pair preparation](../../src/molgap/k1_pair_preparation.py),
`prepare-workflow`, the registered family trainer and the
`kaggle-molgap-workloads` platform skill. Do not import the old experiment CLI
or duplicate preparation/training plumbing. Parent binds source, recipes,
prospective records and the explicit release before publication; these docs
are not a launch receipt or evidence that any remote work occurred.

The training authorization grants no automatic retry,500K/full
run, protected-role access, model promotion or server takeover.
