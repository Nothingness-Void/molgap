# Expert Oracle feasibility

Desktop saved-prediction diagnostic; no remote training or inference.

- [Paper review](paper_review.md): four full-text studies and two abstract-only leads.
- [Frozen protocol](protocol.md): diagnostic estimands, folds and nomination rule.
- [Results and attribution](decision.md): Oracle versus realizable gains and limitations.
- [Conditional experiment plan](experiment_plan.md): prerequisites and two specialist mechanisms.
- [Canonical analysis](analysis.json), [acceptance](acceptance.json), and
  [terminal RML evidence](rml/rml_finalized/v5_evidence.json).

The five prediction payloads and OOF outputs are minimally retained under
platforms/_records/local/expert_oracle_feasibility_20260930; analysis.json binds
their exact hashes. Reproduce with the project Python and PYTHONPATH for the
selected checkout; analyze.py refuses to overwrite an existing result. A new
analysis after pretraining is a separate prospective question, not a silent rerun.
