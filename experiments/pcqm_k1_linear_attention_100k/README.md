# Node-specific linear global exchange

Question: can query-dependent, normalized kernel communication replace K1's
single-slot bottleneck without dense atom-pair attention or 3D?

- [Protocol and literature/negative-history attribution](protocol.md)
- [Exact scientific and cost contract](training_contract.json)
- [Operational status](STATUS.md)
- [Terminal decision, matched trajectories and portability](decision.md)
- Source: `src/molgap/k1_linear_attention.py`; shared training remains in
  `src/molgap/pcqm_k1_variants_runner.py`.
- `freeze_release.py` freezes reference-bound prelaunch and native RML plans;
  `package_source.py` verifies the immutable K1 audit inputs.
- `accept.py` verifies saved outputs only. `prepare_rml_terminal.py` binds the
  native training trace separately from failed/recovered audit identities;
  `results/rml_closure.json` points to finalized and replay-verified evidence.

No full-scale, extra seed, automatic successor, or protected-role authority.
