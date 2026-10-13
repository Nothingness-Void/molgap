# Fixed graph input pipeline diagnostic

Execution-only local comparison of the existing fixed-index baseline,
block-local shuffle, and random-window size buckets. No model promotion.

- [Protocol](protocol.md) owns scope and limits.
- [Preparation](prepare.py) freezes existing source/data/init identities and
  publishes the prospective record through `research_memory.plan`.
- [Local entry](run.py) calls the shared diagnostic, not another trainer.
- `results/result.json` owns measurements after execution.
- `terminal_decision.md` will own terminal interpretation.

Reusable implementation: `src/molgap/input_pipeline_diagnostic.py`.
Live family recipes retain their original samplers. Platform submission and
protected-role access are outside this experiment.
