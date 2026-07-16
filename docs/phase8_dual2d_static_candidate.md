# Phase 8 Dual-2D Static Candidate

## Status

This is an active candidate experiment, not a production model version. It is
therefore deliberately not assigned a later mainline version number.
The current production B3LYP accuracy predictor remains routed dual-GPS v4.

## Hypothesis

After removing the weak Geometry expert, independently trained Local GINE6 and
Global GPS9 may provide stable complementary errors. The 30k pilot uses a
scaffold-disjoint split with complete Local/GPS stacks for seeds 42, 43, and 44.

## Gate Result

Target-wise static blending improves each seed's best single-expert Gap MAE by
`0.001303/0.012029/0.003400` eV. Equal averaging, embedding concat Fusion, and
both dynamic gates fail the all-seed direction rule. The candidate is therefore
limited to static target-wise blending.

## Next Gate

Evaluate the frozen three-seed static blends on common, OOD, P8-hard, and
PCQM-like distributions. Do not open the sealed sets or allocate a production
version unless transfer direction is stable. The three-expert precursor is a
closed archive-r03 experiment; its evidence is in
`results/phase8/archive/archive-r03-three-expert-moe/pilot_decision.md`.

Exact candidate record:
`results/phase8/dual2d_static_candidate/dual2d_decision.md`.
