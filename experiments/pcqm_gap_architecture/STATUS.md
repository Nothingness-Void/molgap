# PCQM Gap Architecture Screen Status

- No molecular-research-server access is authorized during architecture
  selection.
- No official PCQM4Mv2 validation or test-dev row is authorized during the
  Kaggle screen.
- Kaggle2 CPU kernel
  `kaseichou/molgap-official-pcqm-gap100k-r1-prep`, version 1, completed with
  a retained acceptance failure because RDKit was absent; its output and log
  remain under `platforms/_records/kaggle/training/pcqm_gap100k_r1_prep_v1`.
- Version 2 is running with the infrastructure-only fix `rdkit==2025.3.5`,
  using the same source commit and official-train-only contract.
- A GPU screen may start only after that cache passes local no-inference
  acceptance.

The GPU seed-42 screen is staged but not submitted. Task order remains
authoritative in `ROADMAP.md`; retrieval hashes will be added only after local
no-inference acceptance.
