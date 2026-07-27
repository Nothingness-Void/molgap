# archive-r02 Late Soft Blend Decision

## Protocol

- 49,879 existing Base/Expert prediction pairs; no new GNN inference for labels.
- Five-fold scaffold-disjoint OOF on 45,478 development molecules.
- Independent 4,401-molecule dev-test; sealed random/hard sets remain unopened.
- Selection target: Gap MAE. Promotion gate: at least 0.001 eV Gap improvement over fixed v4.

## Results

| Model | HOMO MAE | LUMO MAE | Gap MAE |
|---|---:|---:|---:|
| Fixed v4 | 0.108174 | 0.111365 | 0.151390 |
| Late blend | 0.108127 | 0.111054 | 0.150509 |

- OOF-selected alpha method: `{'homo': 'lightgbm', 'lumo': 'lightgbm', 'gap': 'lightgbm'}`.
- OOF-selected output structure: `independent_three_output`.
- Gap improvement: 0.000881 eV; paired bootstrap 95% CI for late-minus-v4 error: [-0.001555, -0.000213] eV.
- Labels satisfy `Gap = LUMO - HOMO` to numerical precision on dev-test (mean absolute residual 0 eV).
- Physics projection is retained as a diagnostic; it is not selected unless it improves OOF Gap MAE.

## Decision

**STOP**. Stop late blending; do not open sealed sets. The production default remains fixed v4.
