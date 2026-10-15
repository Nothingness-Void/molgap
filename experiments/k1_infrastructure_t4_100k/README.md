# K1 infrastructure execution comparison

<!-- traceability-role: diagnostic -->

Compare the retained original K1 trainer with the optimized shared trainer at
fixed100K. Both use the single-forward reference model.

Read [protocol](protocol.md), [release declaration](authority.json), and the
dated [preparation decision](decision.md), then the
[terminal result](results/decision_2026-10-15.md). Live operation is routed by CURRENT_STATE.md.

[Exact run and recovery route](STATUS.md). Saved artifacts are compared by
[the shared analysis owner](../../src/molgap/k1_execution_analysis.py); platform
retrieval and both existing family inspectors remain authoritative.

[Deep profiler attribution and code opportunities](results/profiler_deep_analysis.md)
separate measured hot paths from unmeasured implementation candidates.
