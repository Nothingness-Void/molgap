# Repaired-2M Retention-D GPS11-160 Seed 42

## Decision

Reject GPS11-160 as a global replacement, hard-region expert, PCQM expert, or
automatic full-scale Fusion identity path. Preserve its checkpoint and complete
2M embeddings as reproducibility and bounded diversity evidence only.

The result does not prove that eleven GPS layers are intrinsically harmful:
GPS11-160 trained from scratch, while the accepted GPS7 and GPS9 controls used
compatible 1.5M warm starts. However, GPS11 reached its best validation point
at the final epoch only after plateauing near `0.1123 eV`, remained
`0.0078-0.0084 eV` behind GPS7/GPS9 validation, and regressed materially on
every general external domain. More continuation is not justified.

## Fixed Comparison

| Model | Common avg/Gap | OOD avg/Gap | P8-hard avg/Gap | PCQM Gap |
|---|---:|---:|---:|---:|
| routed v4 500K | 0.103654 / 0.122581 | 0.112783 / 0.132652 | 0.094329 / 0.112293 | **0.291691** |
| repaired-2M GPS7 | 0.100074 / 0.116918 | **0.109555 / 0.128555** | 0.090390 / 0.105031 | 0.309216 |
| repaired-2M GPS9 | **0.099599 / 0.116133** | 0.110666 / 0.129820 | **0.088293 / 0.102152** | 0.310251 |
| repaired-2M GPS11-160 | 0.113631 / 0.136122 | 0.118972 / 0.141911 | 0.108176 / 0.130209 | 0.302854 |

GPS11-160 costs `23,311 s`, versus `16,528 s` for GPS7 and `19,706 s` for
GPS9. It uses 3.17M parameters and requires one encoder pass.

An evaluation-only target-specific Oracle shows GPS11 differs from GPS9, but
that ceiling uses true external labels and cannot supply Router training
labels. Because the existing GPS7/GPS9 Oracle already passed with a stronger
base and lower complexity, GPS11 must not expand the OOF plan at this stage.
If full-scale Route B Fusion is revisited after 3D assets exist, first compare a
GPS9 identity path against the original GPS11 identity path on development
evidence; GPS11 may only enter as a bounded residual feature.

## Artifact Acceptance

- Jobs `709534`, `709562`, and `709563` completed with exit code zero.
- Model SHA256:
  `d806a4b6907eafd20dec8c5eeae766d7049a5cf7f159295422f398e588829a6a`.
- Complete embedding shape: `2,000,000 x 160`.
- Remote manifest hashes match the metrics, training state, and embedding.
- The downloaded checkpoint contains 340 finite state tensors.
- Common 1,977-row and PCQM 4,981-row prediction files are complete and finite.
- Stderr contains only the known non-fatal DCU memory-efficient-attention
  warning.

Machine-readable acceptance:
`results/phase8/repaired_2m/gps11_160_seed42_acceptance.json`.
Raw bounded artifacts:
`results/phase8/repaired_2m/gps11_160_seed42_raw/`.
No sealed rows were opened and the production registry was not changed.
