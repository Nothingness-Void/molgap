# MolGap Project Progress Visualization

## Last updated
2026-06-05

This page is a report-style visualization summary of the project up to the current point.

## 1. Master Data / Experiment Flow

```mermaid
flowchart TD
    A["PubChemQC HuggingFace subsets"] --> B1["Phase 1 data\nCHON, MW 200-300"]
    A --> B2["Phase 2 data\nexpanded chemistry, 10k/step"]
    A --> B3["Phase 3/4 data\nCHONSFCl, MW 200-500, 30k"]

    B1 --> C["clean.py\ncanonicalize + dedupe + gap check"]
    B2 --> C
    B3 --> C

    C --> D["features.py\nMorgan + MACCS + AtomPair + Torsion + RDKit desc"]
    D --> E1["Phase 1\nbaseline / tuning / embeddings / advanced models"]
    D --> E2["Phase 2\ngeneralization study"]
    D --> E3["Phase 3\nfeature selection + Optuna optimization"]

    C --> F["Phase 4 3D branch\nRDKit ETKDG conformers"]
    F --> G["GNN models\nAttentiveFP / SchNet 3D"]

    E1 --> H1["results/phase1/"]
    E2 --> H2["results/phase2/generalization/"]
    E3 --> H3["results/phase3/ and results/phase3/optimize/"]
    G --> H4["results/phase4/"]

    H3 --> I["Current best traditional model\nTuned LightGBM"]
    H4 --> J["Current best overall model\nSchNet 3D"]

    I --> K["Phase 5 / 6\ncommercial prediction + database (deferred)"]
    J --> K
```

## 2. Phase Roadmap

```mermaid
flowchart LR
    P1["P1\nCHON baseline + tuning + embeddings\nBest: LGBM tuned\nR² 0.9205"] --> 
    P2["P2\nGeneralization study\nR² 0.901 → 0.874"] --> 
    P3["P3\nCHONSFCl 30k + feature selection + Optuna\nBest traditional: LGBM tuned\nR² 0.8853"] --> 
    P4["P4\nEnsemble + GNN\nBest overall: SchNet 3D\nR² 0.8942"] --> 
    P5["P5\nCommercial prediction\nDeferred"] --> 
    P6["P6\nDatabase construction\nDeferred"]
```

## 3. Phase Summary Table

| Phase | Main work | Data scope | Best result | Key conclusion |
|---|---|---|---|---|
| P1 | Baseline, tuning, embeddings, advanced models | 30k CHON, MW 200-300 | Tuned LightGBM, MAE 0.1498, R² 0.9205 | Traditional 2D features are very strong on easier chemistry |
| P2 | Generalization study | 10k per step, expanded chemistry | Step0 baseline R² 0.9012, step4 R² 0.8736 | Performance declines smoothly as chemistry broadens |
| P3 | CHONSFCl scale-up + feature selection + Optuna | 30k CHONSFCl, MW 200-500 | Tuned LightGBM, MAE 0.1596, R² 0.8853 | Better 2D fingerprints + feature selection help, but not enough to reach 0.9 |
| P4 | Ensemble + GNN | 30k CHONSFCl, MW 200-500 | SchNet 3D, MAE 0.1492, R² 0.8942 | 3D geometry is the first clear win over the best LightGBM |
| P5 | Commercial prediction | Application stage | Script ready | Deferred until model/report side is stable |

## 4. Chart Files

- `results/overview/phase2_generalization_curve.png`
- `results/overview/hard_task_progress.png`
- `results/overview/model_family_snapshot.png`
- `results/overview/phase_summary.csv`

## 5. Current headline messages

### If you want the best traditional model
- Use `Phase 3 tuned LightGBM`
- Best hard-task traditional result: `avg MAE=0.1596`, `avg R²=0.8853`

### If you want the best overall model
- Use `Phase 4 SchNet 3D`
- Best hard-task overall result: `avg MAE=0.1492`, `avg R²=0.8942`

### If you want the cleanest one-line conclusion
- The project has progressed from a strong 2D fingerprint baseline to a 3D GNN that closes most of the remaining gap to `R²=0.9` on the hardest current chemistry setting.
