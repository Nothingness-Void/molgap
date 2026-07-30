# GPS7 + SchNet 500K V3

This directory contains the training and evaluation evidence for the 500K v3
single-hybrid component:

- 2D encoder: GPS7, width 192.
- 3D encoder: SchNet, width 192.
- Fusion: gated GPS7 + SchNet head.
- Training set: expansion 500K.
- Role: compatibility/component loader used by routed v4.

The retired 300K v2 evidence is in
`../_retired/gps7_schnet_300k_v2/`. Cross-version comparison tables are in
`../../04_evaluate/model_versions_v1_v2_v3/`.

Large frozen embeddings are stored under
`data/cache/production/gps7_schnet_500k_v3/embeddings/`.
