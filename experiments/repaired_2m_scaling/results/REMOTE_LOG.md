# Repaired-2M scaling — completed remote rounds

Dated record of finished SCNet rounds for this experiment. Each entry states what
was measured and links its decision; none of these changed the production
registry, and no sealed-20K rows were used.

## Three-GPS embedding residual head (accepted, internal gate only)

GPS7/GPS9 each produced 40 contiguous hash-verified 50K-row embedding parts. The
three `+-0.10 eV` residual-head seeds improve the fixed GPS7+GPS9 equal identity
on internal exact-2M test by a mean `0.001821 eV` average MAE and `0.002062 eV`
Gap MAE; all targets improve for all seeds, with an average-MAE seed standard
deviation of `0.000012 eV`. This is the internal gate only — the three-pass
candidate still needs frozen common/OOD/P8-hard evaluation and compute-cost
accounting before any promotion.
Decision: `three_gps_embedding_residual/decision.md`.

## Three-GPS learned routing pilot (Router rejected, blends advance)

Job `709815` passed all holdout, checkpoint, prediction, identity, finite-value,
and SHA256 checks. The learned pre-dispatch Router is **rejected**: it collapses
to GPS9 for every molecule, uses GPS7 only for part of LUMO, and never calls
GPS11-160.

The three-pass dense gate is positive on internal test/common/OOD versus GPS9 by
`0.002828/0.001865/0.003912 eV`; P8-hard average regresses a statistically
inconclusive `0.000226 eV`, driven by `+0.000623 eV` Gap. The robust two-pass
control is a fixed GPS7+GPS9 equal blend: common/OOD/P8-hard average improves by
`0.001044/0.001760/0.000312 eV` versus GPS9. GPS11 contributes useful
correlated-error diversity to dense/mean blending despite a weak standalone
result. For PCQM-valid, dense and equal-three reach `0.302120/0.299602 eV`, still
behind routed v4 and the accepted PCQM GINE expert, so PCQM stays
deterministically specialist-routed.

Advance only the two-pass equal GPS7+GPS9 base and the three-pass dense base to
the bounded dual-SchNet A/B; do not advance the hard Router. This is accepted
pilot evidence, not production registration and not a substitute for formal OOF.
Decision: `three_gps_router_fusion/decision.md`.

## GPS11-160 (rejected as a global replacement)

Jobs `709534`, `709562`, `709563` completed; checkpoint, metrics, predictions,
and the complete `2,000,000 x 160` embeddings passed acceptance. Rejected as a
global replacement, hard expert, PCQM expert, and automatic full-scale Fusion
identity path. Versus repaired-2M GPS9, average MAE regresses by
`0.01403/0.00831/0.01988 eV` on common/OOD/P8-hard; PCQM Gap improves by
`0.00740 eV` but stays `0.01116 eV` worse than routed v4 500K and far behind the
accepted PCQM GINE specialist. GPS11 trained from scratch unlike the warm-started
GPS7/GPS9 controls, but its late plateau and broad external regressions do not
justify continuation. Keep it only as bounded diversity evidence. Any later
Route B pilot must compare a GPS9 identity path against GPS11 identity before
using the full-scale Fusion protocol.
Decision: `gps11_160_seed42_decision.md`.

## Route A first gate (GPS9 hard-expert candidate)

Jobs `709046`/`709047`: repaired-2M GPS9 improves common and P8-hard over
Retention-D GPS7 but regresses OOD and PCQM, so it is rejected as a global
replacement and retained only as a hard-expert candidate. The target-specific
Oracle then passed at a 10% GPS9 call budget, authorizing scaffold-disjoint OOF
gain-label generation but not Router training.
Decisions: `gps9_seed42_decision.md`, `gps7_gps9_oracle_20260725/decision.md`.

## Retention-D three-seed gate (accepted general base)

Retention-D passed against retention-B. Mean common/OOD/P8-hard average-MAE
improvements are `0.001217/0.001496/0.000932 eV`, and every domain improves for
each of seeds 42, 43, 44. PCQM Gap regresses `0.001058 eV` on average and stays a
separately routed specialist domain. Seed 43/44 models and artifacts were
retrieved with matching remote/local SHA256 and finite predictions. Keep seed 42
as the single-pass general base; the repeat seeds are stability evidence, not an
automatic deployment ensemble.
Decision: `retention_d_multiseed_decision.md`.
Manifest: `retention_d_experiment_manifest.json`.

## Retention-aware exact-2M GPS7 controls (B accepted, C cancelled)

Existing uniform exact-2M is control A. B (`705497` -> `705498`) improved
common/OOD/P8-hard average by `0.00242/0.00204/0.00280 eV` but regressed PCQM Gap
by `0.01702 eV`, failing the global gate. C cached all 500K teacher targets then
hit an FP16/FP32 assignment error before training; the error was fixed, but once
B was accepted for common/OOD/P8-hard and PCQM was split into a separate
specialist, replacement jobs `706141` -> `706142` were deliberately cancelled
before any completed epoch to avoid wasting card hours.
Configuration and gates: `../retention_2m_scnet/experiment_manifest.json`.

## Independent artifact acceptance

Job `704402` passed all model, prediction, embedding-part, Parquet-part,
finite-value, row-accounting, uniqueness, and SHA256 checks.
Record: `platforms/_records/scnet/overnight_20260723_acceptance.json`.

## P8.19 chain

Graph construction, GPS7/GPS9, dual-2D head, development evaluation,
frozen-embedding staging, and graph-cache archival all completed. The verified
staging payload is the private Kaggle dataset
`nothingnessvoid/molgap-2m1m-fusion-staging-20260722`.
Local handoff: `../../multi2d_experts/multi2d_2m_hard20k/`.
