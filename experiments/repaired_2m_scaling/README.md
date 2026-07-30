# Repaired-2M Scaling

**Question:** Does scaling the repaired corpus to two million molecules improve
the general model, and can bounded 3D residuals add value?

**Verdict:** Retention-D is the accepted 2D base; ordinary fusion is rejected.
The bounded 3D path remains gated by accepted SchNet checkpoints and embeddings.

Read `STATUS.md` for live remote state, `results/decision.md` for the data gate,
and `scaleup_full_analysis/decision.md` for unified scale-up evidence.
