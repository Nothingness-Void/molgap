# MolGap Roadmap — Requirements & Checklist

## Final Goal

Train a model on PubChemQC (B3LYP/6-31G*) data, then **batch-predict HOMO / LUMO / Gap for commercially available molecules** used in OLED, organic thin-film, and organic solar cell research.

### Primary deliverable
**Molecular property database** — a curated table of commercial organic electronic molecules with predicted HOMO/LUMO/Gap (eV), ready for researchers to query.

### Secondary deliverable (tentative)
**Paper / report** — if required by advisor, write up the methodology and experimental findings. Not yet confirmed.

---

## Deliverable 1: Molecular Property Database (PRIMARY)

End goal: a CSV/Excel of commercial molecules with columns `name, supplier, SMILES, HOMO, LUMO, Gap, confidence`.

### Pipeline components

| # | Item | Status | Notes |
|---|------|--------|-------|
| D1 | Best trained model | done | `models/gnn_schnet_3d_optuna_expanded.pt` (Phase 6, R²=0.882) |
| D2 | Batch inference script (SMILES list → predictions) | **TODO** | SMILES → ETKDG → graph → model → CSV |
| D3 | Curated commercial molecule list | **TODO** | OLED / thin-film / OPV molecules from TCI, Sigma-Aldrich, Ossila, etc. Template exists |
| D4 | Final database generation (D2 + D3) | **TODO** | Run batch prediction on curated list |
| D5 | Confidence / uncertainty per prediction | **TODO** | Conformer ensemble std or applicability domain flag |

### Model improvement (before final database)

| # | Item | Status | Notes |
|---|------|--------|-------|
| D6 | Hybrid 2D+3D experiment result | **running** | Kaggle, may replace D1 if better |
| D7 | 300k data scaling + retrain | **TODO** | Highest expected impact on OOD accuracy |

### Nice-to-have

| # | Item | Status | Notes |
|---|------|--------|-------|
| D8 | Web UI (Gradio/Streamlit) for single-molecule query | **TODO** | |
| D9 | Environment reproducibility (`requirements.txt` with pinned versions) | partial | Missing torch/pyg versions |

---

## Deliverable 2: Paper / Report (TENTATIVE)

If advisor requires it, the experimental record from Phase 1-7 is the basis. All key experiments are already done — this is mostly a writing/figure-generation task.

### Experimental results available

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Traditional ML baseline (LightGBM R²=0.921) | done |
| 2 | Generalization study (element/MW expansion) | done |
| 3 | Scaled ML optimization (30k, R²=0.885) | done |
| 4 | SchNet GNN + train-inference consistency | done |
| 5 | OOD + Gaussian + experimental validation | done |
| 6 | MW expansion (44.8k, R²=0.882, Gaussian Gap MAE=0.223) | done |
| 7 | Conformer ensemble / Hybrid 2D+3D / 300k scaling | in progress |

### Figures/tables to generate when needed

- Parity plots per target per phase
- R²/MAE progression across all phases
- B3LYP vs experimental bias analysis
- Conformer noise quantification
- Training data size vs performance curve
- Model architecture diagram

---

## Phase Summary & Classification

| Phase | Category | Purpose | Status |
|-------|----------|---------|--------|
| 1 | Baseline | Traditional ML upper bound | done |
| 2 | Analysis | Generalization / domain expansion feasibility | done |
| 3 | Baseline | Scaled traditional ML upper bound | done |
| 4 | Core model | GNN with train-inference consistency | done |
| 5 | Validation | External validation (OOD, Gaussian, experiment) | done |
| 6 | Scaling | MW expansion + retuning | done |
| 7 | Improvement | Conformer ensemble / 2D+3D hybrid / data scaling | **in progress** |

### Phase 7 sub-experiments status

| Experiment | Status | Result |
|------------|--------|--------|
| Conformer ensemble (ETKDG, k=1→8) | done | +2.5% R², marginal |
| xTB conformer replacement | suspended | Low priority, marginal expected gain |
| Hybrid 2D+3D SchNet | running | On Kaggle, awaiting results |
| 300k data scaling | TODO | Script ready, highest expected impact |

---

## Current Bottlenecks (from Phase 6 analysis)

1. **ETKDG conformer noise** — ~0.08-0.10 eV std per molecule (Phase 7 quantified). Ensemble averaging helps marginally.
2. **Training data coverage** — 44.8k covers tiny fraction of PubChemQC 85M. Scaling to 300k is the most promising path.
3. **B3LYP label accuracy** — Kohn-Sham orbital energies have known systematic bias vs experiment. This is a fundamental ceiling, not fixable by ML.

---

## Priority Order for Remaining Work

| Priority | Task | Deliverable | Expected Impact | Effort |
|----------|------|-------------|-----------------|--------|
| 1 | Get Hybrid 2D+3D results from Kaggle | D6 | unknown, may improve model | waiting |
| 2 | 300k data scaling + retrain | D7 | high (OOD R² 0.80→0.88+) | 2-3 days Kaggle |
| 3 | Curate commercial molecule list | D3 | required for database | 1-2 days |
| 4 | Build batch inference pipeline | D2 | required for database | 0.5 day |
| 5 | Generate final database | D4 | **primary deliverable** | 0.5 day |
| 6 | Uncertainty estimation | D5 | adds credibility | 1 day |
| 7 | Paper figures/writing | Deliverable 2 | if advisor requires | 1-2 days |
| 8 | Web UI | D8 | nice-to-have | 0.5-1 day |
