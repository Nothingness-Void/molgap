# MolGap TODO

## Last updated
2026-06-05

## Completed phases
- [x] Phase 1: CHON baseline, tuning, embeddings, advanced-model comparison
- [x] Phase 2: generalization study across broader chemistry
- [x] Phase 3: CHONSFCl 30k scale-up, feature selection, and model optimization
- [x] Phase 4: ensemble and GNN experiments

## Active priority

### P0 — Documentation sync
- [x] Bring active TO-AI docs in line with the current phase-based repo structure.

### P1 — Final model/report closeout
- [ ] Consolidate the final model comparison into one report-ready summary.
- [ ] Decide which result is the main final benchmark:
  `Phase 3 tuned LightGBM` or `Phase 4 SchNet 3D`.
- [ ] Decide whether the final gap result should report direct gap, blended gap, or both.
- [ ] Update any remaining stale docs or command references after the final benchmark decision.

### P2 — Optional accuracy push
- [ ] Try one more SchNet-focused tuning round if the goal is to cross `R²=0.9` on CHONSFCl 30k.
- [ ] Decide whether expanding the hard chemistry dataset beyond `30k` is worth the cost.
- [ ] If needed, run one more controlled comparison between:
  `SchNet 3D`, `ridge stack`, and `Phase 3 tuned LightGBM`.

### P3 — Repository cleanup
- [ ] Decide which root-level `results/` outputs are legacy and can be archived or deleted.
- [ ] Keep `results/phase*/` and `results/common/` as the canonical output structure.
- [ ] Sync `requirements.txt` with the actually used packages if you want fully reproducible setup docs.

## Current best results

### CHON, MW 200-300
- [x] Tuned LightGBM on `30k` reached `avg MAE=0.1498`, `avg R²=0.9205`.

### CHONSFCl, MW 200-500
- [x] Phase 3 tuned LightGBM reached `avg MAE=0.1596`, `avg R²=0.8853`.
- [x] Phase 4 ridge stacking reached `avg MAE=0.1543`, `avg R²=0.8912`.
- [x] Phase 4 SchNet 3D reached `avg MAE=0.1492`, `avg R²=0.8942`.

## Deferred application work

### Phase 5 — Commercial prediction
- [x] Commercial prediction script exists:
  `scripts/phase5/predict_commercial.py`
- [x] Template input exists:
  `data/commercial/commercial_molecules_template.csv`
- [x] Smoke-test output exists:
  `results/phase5/database/commercial_molgap_predictions_v1.csv`
- [ ] Curate real supplier molecules.
- [ ] Run prediction on the real commercial list.
- [ ] Filter low-confidence / out-of-domain molecules.

### Phase 6 — Database construction
- [ ] Merge final predictions and metadata into one final database table.
- [ ] Add source labels:
  `PubChemQC calculated` vs `ML predicted`.
- [ ] Select molecules for possible Gaussian verification.
- [ ] Export final database + documentation.

## Canonical references
- [ ] Keep `TO-AI/experiment_phases.md` as the high-level phase summary.
- [ ] Keep `results/master_experiment_log.csv` as the canonical experiment table.
- [ ] Keep `TO-AI/handover.md` as the short current-state summary for the next session.
