# Status

The architecture and v4 execution contract were released on Kaggle2 as:

- `kaseichou/molgap-pcqm-k1-v4-reference-s42`, version 2, one P100, K1-v4;
- `kaseichou/molgap-pcqm-k1-v4-candidates-s42`, version 2, T4x2, K1-G and K1-R.

Both repaired jobs reached `RUNNING` on 2026-09-12. Candidate version 1 and
reference version 1 stopped before importing model code because Kaggle expanded
`src.zip` into a dataset directory; neither performed training. The
source-discovery repair preserves the scientific contract. No scientific result
exists until both version-2 outputs pass `accept.py` without model inference.
