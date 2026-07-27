# Repair-v2 Pure 2D Control Evaluation

This kernel compares the completed controlled dual-GPS 1M heads on fixed labels:
the shared common/OOD/P8-hard set and the fixed 5,000-row PCQM4Mv2 validation
sample. It does not construct ETKDG coordinates, load SchNet, or use a 3D
fusion head.

Inputs are the v1 1M GPS7/GPS9 checkpoints already in
`molgap-1m-external-eval-models`, plus four appended repair-v2 artifacts:
GPS7, GPS9, and the two controlled dual-GPS heads. The runtime writes a compact
progress JSON and saves each completed label set independently before moving to
the next one.

The decision is based only on paired external deltas and bootstrap intervals.
The model-specific internal holdouts are descriptive, not an acceptance
criterion, because their data distributions differ.
