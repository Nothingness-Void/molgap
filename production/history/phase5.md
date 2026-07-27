# Phase 5: Validation (OOD + Gaussian + Experimental)

> Historical method and evidence. Live project state is in `CURRENT_STATE.md`.

## Goal
Validate the Phase 4 model on external data: OOD PubChemQC molecules, Gaussian B3LYP single-point calculations, and experimental literature values.

## Results

### Commercial OLED molecules (10)
- 10/10 ETKDG conformer generation succeeded
- 6/10 have MW > 500 (extrapolation for Phase 4 model)

### OOD validation (100 PubChemQC molecules)
- avg R²=0.849, MAE=0.188

### Experimental comparison (9 molecules)
- HOMO: B3LYP systematically shallower by ~0.5-0.7 eV (Koopmans approximation)
- LUMO: B3LYP shallower by ~1.3-2.1 eV (known DFT virtual orbital deficiency)
- Linear correction unreliable (only 9 data points, narrow scaffold diversity)

## Key findings
- B3LYP Kohn-Sham energies have systematic bias vs experiment — this is a label-level limitation, not fixable by ML
- MW>500 molecules are extrapolation → motivates Phase 6 MW expansion

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/phase5/gaussian_validation.py` | Gaussian B3LYP comparison |
| `scripts/phase5/parse_gaussian_outputs.py` | Parse Gaussian log files |
| `scripts/phase5/ood_validation.py` | OOD 100-mol evaluation |

## Results
`results/phase5/`

## Dependencies
- Phase 4 model (`models/gnn_schnet_3d_tuned.pt`)
- `data/commercial/` for OLED molecule SMILES
