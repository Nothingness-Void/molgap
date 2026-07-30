# Repaired-2M Hierarchical Dual-SchNet Fusion

## Decision

The predeclared bounded 2D+3D residual gate passed its internal
scaffold-disjoint acceptance test for both frozen repaired-2M 2D identities.
The dense 2D identity produced the best absolute fused result, while the equal
GPS7/GPS9 identity obtained the larger improvement from 3D.

This was development evidence, not a production promotion. The later paired
common/OOD/P8-hard evaluation rejected both hierarchical dual-SchNet residual
variants because they regressed against their own frozen 2D identities. The
pure-2D identities passed against routed-v4 500K. See
`../hierarchical_dual_schnet_external/decision.md`.

## Internal Test

The accepted aligned subset contained 198,932 molecules. Scaffold groups were
split into 160,057 training, 19,501 validation, and 19,374 test rows with zero
scaffold overlap. Each result below is a three-seed ensemble of bounded
residual heads using the accepted Primary and Augmented SchNet embeddings.

| Frozen 2D identity | Base average MAE | Fused average MAE | Delta |
|---|---:|---:|---:|
| Equal GPS7/GPS9 | 0.106114 | 0.102012 | -0.004102 |
| Dense | 0.104361 | **0.101906** | -0.002455 |

All values are in eV. Both average-MAE bootstrap intervals were strictly below
zero:

- Equal GPS7/GPS9: `[-0.004433, -0.003766] eV`
- Dense: `[-0.002725, -0.002180] eV`

All three targets improved for both identities. For the dense identity, fused
HOMO/LUMO/Gap MAE was `0.092076/0.091621/0.122021 eV`.

## Artifact Acceptance

- IMS preflight job: `1118215`
- IMS training job: `1118217`
- Six best heads and six resumable last checkpoints: finite and contract-valid
- Remote/local SHA256: identical for `metrics.json` and all six best heads
- Metrics SHA256:
  `31018796e407fbcfed65b1061f9d779c5bad6a54bd6e60b2d6759df27bc74fd9`
- Production registry changed: no
- Sealed 20K used: no

The raw accepted output is retained in
`../hierarchical_dual_schnet_v1_remote/`.
