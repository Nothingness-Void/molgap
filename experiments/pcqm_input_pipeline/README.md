# Fixed graph input pipeline diagnostic

Execution-only local comparison of the existing fixed-index baseline,
block-local shuffle, and random-window size buckets. No model promotion.

- [Protocol](protocol.md) owns scope and limits.
- [Preparation](prepare.py) freezes existing source/data/init identities and
  publishes the prospective record through `research_memory.plan`.
- [Local entry](run.py) calls the shared diagnostic, not another trainer.
- [Measurements](results/result.json) own the executed timings and identity.
- [Terminal decision](terminal_decision.md) owns interpretation and disposition.
- [Acceptance](acceptance.json) and canonical `rml/rml_finalized/` own closure;
  this diagnostic is not training replay-ready or a quality nomination.

Reusable implementation: `src/molgap/input_pipeline_diagnostic.py`.
Live family recipes retain their original samplers. Platform submission and
protected-role access are outside this experiment.
