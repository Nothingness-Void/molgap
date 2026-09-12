# Status

The architecture and v4 execution contract were released on Kaggle2 as:

- `kaseichou/molgap-pcqm-k1-v4-reference-s42`, version 2, one P100, K1-v4;
- `kaseichou/molgap-pcqm-k1-v4-candidates-s42`, version 2, T4x2, K1-G and K1-R.

Both repaired jobs completed on 2026-09-12 and passed joint no-inference
acceptance. Candidate version 1 and reference version 1 stopped before importing
model code because Kaggle expanded `src.zip` into a dataset directory; neither
performed training. The source-discovery repair preserved the scientific
contract. `decision.md` closes K1-G and K1-R without a selected candidate.
