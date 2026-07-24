# Phase 2: Generalization Study

> Historical method and evidence. Live project state is in `CURRENT_STATE.md`.

## Goal
Test how model performance degrades when expanding element types and MW range.

## Data
- Fixed: LightGBM (Phase 1 tuned params), 10k per step
- Steps: progressively add elements and widen MW

## Results
| Step | Elements | MW | R² |
|------|----------|-----|-----|
| 0 | CHON | 200-300 | 0.901 |
| 1 | CHON | 200-500 | 0.889 |
| 2 | CHONS | 200-500 | 0.879 |
| 3 | CHONSF | 200-500 | 0.878 |
| 4 | CHONSFCl | 200-500 | 0.874 |

## Key findings
- R² decays smoothly (no cliff) — safe to expand
- HOMO most sensitive to chemical diversity, LUMO most stable

## Scripts
`scripts/phase2/generalization_study.py`

## Results
`results/phase2/generalization/`

## Dependencies
- Uses Phase 1 tuned LightGBM params
- Fetches fresh data per step via pipeline
