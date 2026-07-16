# Phase 8 Dual-2D Static Candidate

This is the only active Phase 8 candidate route, not a production model version.
A 30k scaffold-disjoint
pilot found that target-specific static blending of Local GINE6 and Global GPS9
improved Gap MAE over the best single expert in all three complete model seeds.
Dynamic gates and embedding concat Fusion did not pass.

Current action: evaluate the three static blends on common, OOD, P8-hard, and
PCQM-like distributions before any larger training run. Sealed archive-r02 sets remain
unopened, and routed dual-GPS v4 remains the production recommendation.

See [`dual2d_decision.md`](dual2d_decision.md) for the gate and exact summary.
