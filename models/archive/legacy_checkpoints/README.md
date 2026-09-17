# Legacy Checkpoints

These local-only weights are retained for reproducibility of closed V1-V3,
retired V2/V3, TensorNet, and historical Delta/UQ work. They are not registry
defaults and must not be loaded by filename guess.

The authoritative historical decisions remain on the `archive` branch. The
original frozen filenames are retained; their mapping is centralized in
`src/molgap/constants.py` and the model registry. New work must use a named
experiment checkpoint under `experiments/<question>/results/` or a registered
asset under `models/phase8/`.
