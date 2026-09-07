# ComponentState seed-43 confirmation status

- Stage: complete, mechanically accepted and scientifically decided.
- Question: paired seed-43 replication of descriptor-only control versus
  persistent ComponentState on T4x2.
- Protocol: `../../component_state_seed43_protocol.md`.
- Frozen data: official-PCQM-train-derived 100,000 train / 10,000 internal
  validation; official validation and test-dev remain unread.
- No GPU submission is allowed until the CPU cache passes no-model acceptance.
- A terminal GPU result authorizes analysis only; seed 44, full-data training,
  official-role access and molecular-research-server access remain locked.
- Source dataset:
  `nothingnessvoid/molgap-pcqm-componentstate-confirm-source`, version 1,
  marker `c3c54e09c783d8718cb3129b08e33fd8de3525cd`.
- CPU cache kernel:
  `nothingnessvoid/molgap-pcqm-componentstate-cache-s43`, version 2.
- Version 1 failed before reading graph data because Kaggle expands uploaded
  `src.zip` into a source tree. Version 2 accepts either representation; the
  cache algorithm, rows and all scientific settings are unchanged.
- Version 2 completed in `310.12 s`. No-model acceptance verified 100,000
  train and 10,000 internal-validation graphs across 22 shards. Accepted
  aggregate SHA-256:
  `434dc60b40064eb30eab279e85dcc7f43ef7763cda9e9e42d7a8630f35de57fc`.
- GPU kernel:
  `nothingnessvoid/molgap-pcqm-componentstate-confirmation-s43`, version 1.
  Kaggle normalized the originally requested shorter slug at submission; the
  tracked metadata now matches the canonical remote identity.
- Seed-43 paired result: descriptor control `0.1305245310 eV`, ComponentState
  `0.1298854351 eV`, delta `-0.0006390959 eV`. The direction replicated, but
  the frozen `0.001 eV` material-gain threshold did not.
- Across seeds 42 and 43 the mean paired delta is `-0.0008335730 eV`.
  ComponentState is retained as weak positive mechanism evidence, not promoted
  over the simpler three-seed GraphState9 handoff. No seed 44 or successor was
  submitted.
- Decision: `decision.md`; compact terminal evidence: `seed43_summary.json`.
