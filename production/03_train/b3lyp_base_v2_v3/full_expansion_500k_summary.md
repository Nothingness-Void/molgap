# Phase 8 Expansion500k Summary

Date: 2026-06-28

## Decision status

`phase8_expansion_hybrid` is a **v3 candidate** registered for explicit inference
and evaluation. It is not the default loader yet; `load_hybrid()` still defaults
to the selected v2 `phase8_replacement_hybrid` until downstream Delta/UQ assets
are revalidated or the default is intentionally switched.

## Dataset

The 500k set keeps all 300,000 rows from `data/raw/phase8_replacement_300k.csv`
and appends 200,000 non-duplicate rows:

- targeted hard/top-up rows: large MW, S/Cl, aromatic edge, flexible hard
- general in-domain rows to fill the remaining quota
- element/MW/label filters inherited from Phase 8
- no duplicate canonical SMILES in the final 500k CSV

Assembly report: `results/phase8/expansion_500k_report.md`.

## Graph caches

- 2D bond graphs: 500,000 / 500,000 succeeded,
  `results/phase8/pyg_2d_graphs_bond_expansion_500k.pt`
- 3D ETKDG graphs: 497,578 / 500,000 succeeded,
  `results/phase8/pyg_3d_graphs_etkdg_expansion_500k.pt`

The 3D failures are expected for a small fraction of larger/heteroatom-rich
molecules under ETKDG. Training and inference both use ETKDG.

## Training

Warm-started from the selected v2 replacement300k checkpoints.

| Component | Model | Best val MAE | Test avg MAE | Test Gap MAE |
|---|---:|---:|---:|---:|
| GPS 2D | `models/phase8_gps_expansion_500k.pt` | 0.11069 | 0.11127 | 0.13400 |
| SchNet 3D | `models/phase8_schnet_expansion_500k.pt` | 0.11800 | 0.11921 | 0.14408 |
| Fusion | `models/phase8_hybrid_fusion_expansion_500k.pt` | 0.08525 | 0.08632 | 0.10043 |

SchNet was trained for 12 warm-start epochs because a full 50-epoch cap would be
9-hour scale on the RTX 5060. Postprocessing uses batch size 128; the previous
256-batch SchNet eval/extract path was too close to the 8 GB VRAM limit.

## Common eval

Common valid N = 1,977 (`ood1000`: 999, `p8_targeted_hard`: 978).

Hybrid MAE:

| Model | All avg | All Gap | OOD avg | OOD Gap | P8 hard avg | P8 hard Gap |
|---|---:|---:|---:|---:|---:|---:|
| Phase 7 full | 0.14529 | 0.17930 | 0.12431 | 0.14881 | 0.16671 | 0.21045 |
| Replacement300k full | 0.12838 | 0.15609 | 0.12144 | 0.14478 | 0.13548 | 0.16765 |
| Expansion500k full | 0.10560 | 0.12528 | 0.11373 | 0.13399 | 0.09729 | 0.11638 |

Expansion500k delta vs Phase 7:

- all avg / Gap: -0.03969 / -0.05402 eV
- OOD-1000 avg / Gap: -0.01058 / -0.01482 eV
- P8 targeted hard avg / Gap: -0.06943 / -0.09406 eV

Expansion500k delta vs Replacement300k:

- all avg / Gap: -0.02279 / -0.03081 eV
- OOD-1000 avg / Gap: -0.00771 / -0.01079 eV
- P8 targeted hard avg / Gap: -0.03819 / -0.05126 eV

Artifacts:

- `results/phase8/full_expansion500k_common_eval_metrics.json`
- `results/phase8/full_expansion500k_common_eval_predictions.csv`

## Notes

- This run supports continuing data expansion beyond targeted-only replacement;
  keeping the 300k replay set and appending broader in-domain molecules was
  beneficial on both OOD-1000 and the P8 hard slice.
- Full 500k MoE was not run. The current win is from data coverage + trainable
  encoders + standard FusionHead.
- Phase 9/10 Delta/UQ assets remain invalidated by any Phase 8 base switch and
  must be revalidated before database generation.
