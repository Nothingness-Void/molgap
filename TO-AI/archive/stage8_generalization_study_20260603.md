# Stage 8 — Generalization Study Archive

Archived at: 2026-06-03.

## Purpose

Systematically test how model performance degrades as molecular diversity increases, step by step. Each step adds elements or expands MW range, with 10k molecules and fixed best-known LightGBM params.

## Script

```text
scripts/experiments/16_generalization_study.py
```

## Results

| Step | Elements | MW | n_mol | avg MAE | avg R² | HOMO R² | LUMO R² | Gap R² |
|------|----------|-----|-------|---------|--------|---------|---------|--------|
| 0 | C,H,N,O | 200-300 | 10000 | 0.1617 | 0.9012 | 0.8772 | 0.9369 | 0.8896 |
| 1 | C,H,N,O | 200-500 | 10000 | 0.1628 | 0.8886 | 0.8495 | 0.9305 | 0.8859 |
| 2 | C,H,N,O,S | 200-500 | 10000 | 0.1671 | 0.8789 | 0.8238 | 0.9241 | 0.8887 |
| 3 | C,F,H,N,O,S | 200-500 | 10000 | 0.1732 | 0.8775 | 0.8172 | 0.9312 | 0.8842 |
| 4 | C,Cl,F,H,N,O,S | 200-500 | 10000 | 0.1754 | 0.8736 | 0.8489 | 0.9077 | 0.8641 |

## Key findings

1. **No cliff-edge degradation** — performance decreases smoothly, not catastrophically.
2. **MW expansion (step 0→1)**: R² drops 0.901→0.889, mainly from HOMO (0.877→0.850). LUMO barely changes.
3. **Adding S (step 1→2)**: small drop, HOMO suffers most (0.850→0.824).
4. **Adding F (step 2→3)**: negligible change overall. LUMO slightly improves.
5. **Adding Cl (step 3→4)**: LUMO drops noticeably (0.931→0.908), Gap drops (0.884→0.864).
6. **Total degradation step0→step4**: avg MAE 0.162→0.175 (+8%), R² 0.901→0.874. Manageable.

## Interpretation

The model generalizes reasonably to broader chemical space. HOMO prediction is most sensitive to molecular diversity (electron-rich elements like S affect HOMO more). LUMO is stable until halogen atoms (Cl) are added. Gap performance tracks the weakest of HOMO/LUMO.

For a production model covering OLED/organic electronics materials (which commonly contain S, F, Cl), training on the full CHONSFCl + MW 200-500 space with more data (30k-100k) should recover much of the lost performance.

## Data files generated

```text
data/raw/step0_chon_mw200_300.csv
data/raw/step1_chon_mw200_500.csv
data/raw/step2_chons_mw200_500.csv
data/raw/step3_chonsf_mw200_500.csv
data/raw/step4_chonsfcl_mw200_500.csv
results/generalization/step*_metrics.json
results/generalization/generalization_summary.csv
```

## Recommended next step

Scale up step4 (CHONSFCl, MW 200-500) to 30k-50k molecules and retrain. Expected to recover R²≈0.90+ based on the 10k→30k improvement pattern observed in CHON-only data.
