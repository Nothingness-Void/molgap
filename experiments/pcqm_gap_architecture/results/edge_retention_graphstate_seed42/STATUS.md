# Edge-retention GraphState seed-42 status

- Stage: Kaggle2 paired T4x2 screen running.
- Kernel: `kaseichou/molgap-pcqm-edge-retention-graphstate-s42`, version 1.
- Source dataset: `kaseichou/molgap-pcqm-local-statistics-source`, version 2.
- Frozen source: `4226f819d9e997f3bae9bf2ad2c0f2d517c1ba1c`.
- Geometry cache aggregate SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Comparison: fresh GraphState9 on GPU 0 versus gated persistent-edge
  retention on GPU 1; seed 42 only.
- The first source-dataset upload omitted the `src/` directory because Kaggle
  CLI defaulted to skip directories. Version 2 added the complete source tree
  before the GPU kernel was submitted; no GPU work used version 1.
- Official validation/test-dev remain unread. No extra seed or full-data action
  is authorized.

