# PCQM Gap Architecture Screen Status

- No molecular-research-server access is authorized during architecture
  selection.
- No official PCQM4Mv2 validation or test-dev row is authorized during the
  Kaggle screen.
- Kaggle2 CPU kernel
  `kaseichou/molgap-official-pcqm-gap100k-r1-prep`, version 1, is building the
  sharded graph cache from source commit
  `a67724999dbe145b38c2792b86d4e654f5589a20`.
- A GPU screen may start only after that cache passes local no-inference
  acceptance.

The GPU seed-42 screen is staged but not submitted. Task order remains
authoritative in `ROADMAP.md`; retrieval hashes will be added only after local
no-inference acceptance.
