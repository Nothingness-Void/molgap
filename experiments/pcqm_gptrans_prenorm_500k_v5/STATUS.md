# Status

Frozen source commit: `8428c7b9832e1ad60ce44522184221652d98bd12`.
Source archive SHA-256:
`cc1b72fb360e485c504fafbbc2e4f6d3999be6cebac68c5b0f4d0c8aad03c202`.

Kunshan jobs submitted on 2026-09-18:

| Arm | Preflight | Training | Release rule |
|---|---:|---:|---|
| reference | `122425237` | `122425440` | training has `afterok:122425237` |
| Pair PreNorm | `122425241` | `122425441` | training has `afterok:122425241` |

No scientific result is assumed until both arms pass independent artifact
acceptance and the paired comparison is recomputed without model inference.
