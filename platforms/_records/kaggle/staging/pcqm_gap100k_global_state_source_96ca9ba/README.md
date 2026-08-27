# Recurrent Graph-State Source Upload

This directory records the private Kaggle source dataset used by the recurrent
graph-state seed-42 screen.

- Remote dataset: `kaseichou/molgap-pcqm-gap100k-global-state-source`
- Pinned source commit: `96ca9ba22a021fcdd8fdf8daecfe60fc0878c5c8`
- Uploaded source archive SHA-256:
  `07d3dceb911155866b1717aefcffc029d0f79312f5be3e2ce72bbf6b7a298abc`

`src.zip` is intentionally ignored by Git because it is a mechanical archive
of the pinned tree. Recreate it from the repository root with:

```powershell
git archive --format=zip --output=src.zip 96ca9ba22a021fcdd8fdf8daecfe60fc0878c5c8 src/molgap
```

The private Kaggle dataset reached `ready` before the GPU kernel was submitted.
