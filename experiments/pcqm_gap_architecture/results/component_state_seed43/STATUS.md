# ComponentState seed-43 confirmation status

- Stage: private Kaggle1 CPU component-cache kernel version 2 is running.
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
