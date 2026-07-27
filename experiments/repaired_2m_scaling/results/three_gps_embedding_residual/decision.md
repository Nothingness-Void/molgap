# Repaired-2M Three-GPS Embedding Residual Decision

## Decision

The internal exact-2M gate passes. Keep the bounded residual head as a
candidate for external common/OOD/P8-hard evaluation.

Do not promote it to production yet. The result is an internal split result,
and the head requires all three GPS encoder passes.

## Result

The identity path is the fixed equal average of repaired-2M GPS7 and GPS9.
GPS7, GPS9, and GPS11-160 predictions and embeddings provide context to a
`+-0.10 eV` bounded correction head.

Across seeds 42, 43, and 44:

| Metric | Identity | Residual | Delta |
|---|---:|---:|---:|
| HOMO MAE | 0.096001 | 0.094122 | -0.001878 |
| LUMO MAE | 0.093164 | 0.091642 | -0.001521 |
| Gap MAE | 0.125517 | 0.123455 | -0.002062 |
| Average MAE | 0.104894 | 0.103073 | -0.001821 |

All three seeds improve every target. The seed standard deviation is
`0.000012 eV` for both average and Gap MAE, so the direction is stable.

## Acceptance

- Both 2M embedding exports contain 40 contiguous 50K-row parts.
- All 80 part sizes and SHA256 hashes match their completion manifests.
- Best and resumable last checkpoints for all three seeds load successfully,
  match their recorded hashes, and contain finite model tensors.
- All seeds use the same data/model contract.
- No sealed-20K rows were used and the production registry was not changed.

Exact machine-readable evidence is in `acceptance.json`; downloaded immutable
manifests, metrics, and checkpoints are under `remote/`.

## Next Gate

Run frozen common/OOD/P8-hard evaluation. Promotion requires a useful external
gain without a material P8-hard regression. The three-pass compute cost must be
reported against the two-pass GPS7+GPS9 identity.
