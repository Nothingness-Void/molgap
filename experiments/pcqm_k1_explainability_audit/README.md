# PCQM K1 explainability audit

- `protocol.md` freezes the diagnostic question and stopping rule.
- `package_payloads.py` packages same-contract prediction payloads.
- `run_error_audit.py` is the thin Stage-1 CLI.
- `stage2_protocol.md` freezes the authorized three-round causal funnel.
- `run_causal_audit.py` is the thin frozen-checkpoint Stage-2 CLI.
- `inspect_cache.py` and `inspect_cache.slurm` are one-off schema probes.
- `results/` receives compact accepted evidence and the final decision.
- [Slot-compression reassessment](results/slot_compression_reassessment.md)
  distinguishes measured exchange effects from unresolved information loss.
- [Representation diagnostic design](representation_protocol.md) scopes the
  distinct same-row frozen-checkpoint measurement and unreleased input gates.
- [Terminal workflow](representation/terminal_runbook.md) routes retrieval,
  saved-JSON acceptance, interpretation and NO_TRAIN closure; the thin
  `accept_representation.py` CLI never runs a model.
- [Curve and mechanism review](results/curve_mechanism_review.md) separates
  exposure catch-up, population instability and larger-scale generalization
  deficits, with a hash-bound 200-observation trace extract.
