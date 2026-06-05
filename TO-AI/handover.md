# MolGap Handover

## Last updated
2026-06-05

## Current repo layout
- `scripts/pipeline/` — shared data pipeline and utility builders
- `scripts/phase1/` — CHON / MW 200-300 model optimization
- `scripts/phase2/` — generalization study across broader chemistry
- `scripts/phase3/` — CHONSFCl / MW 200-500 scale-up and optimization
- `scripts/phase4/` — ensemble and GNN experiments
- `scripts/phase5/` — commercial prediction pipeline
- `scripts/colab/` — embedding extraction helpers for Colab
- `src/molgap/utils.py` — shared utilities

## Current project status
The project is no longer in the early baseline stage. It has advanced through:

1. Phase 1: baseline, tuning, embeddings, and advanced-model comparison on CHON
2. Phase 2: generalization study as element set and MW range expand
3. Phase 3: scale-up to CHONSFCl MW 200-500 at 30k molecules
4. Phase 4: deep learning and ensemble experiments
5. Phase 5: commercial prediction script exists, but application work is still deferred

The canonical experiment overview is:

```text
TO-AI/experiment_phases.md
results/master_experiment_log.csv
```

`results/master_experiment_log.csv` currently contains `44` experiment records.

## Important scientific decisions
- Energies are already treated as `eV`; do not apply Hartree-to-eV conversion.
- Fixed split handling is centralized in `src/molgap/utils.py`.
- Commercial prediction is not the active priority; the active priority is model/report completion.
- For the harder chemistry space, `CHONSFCl, MW 200-500`, 3D geometry matters: SchNet 3D currently beats the best fingerprint model.

## Best results by phase

### Phase 1 — CHON, MW 200-300
- Best tuned traditional model on `30k CHON`: LightGBM tuned
- Test result: `avg MAE=0.1498`, `avg R²=0.9205`
- Embeddings did not improve over traditional fingerprints + descriptors

### Phase 2 — Generalization study
- Chemistry expanded from `CHON 200-300` to `CHONSFCl 200-500`
- Performance declines smoothly, not catastrophically
- `avg R²` roughly drops from `0.901` to `0.874`

### Phase 3 — CHONSFCl, MW 200-500, 30k
- Added richer fingerprints:
  `Morgan + MACCS + AtomPair + Torsion + RDKit descriptors`
- Raw features: `6028`
- Gain-selected features: `2811`
- Best traditional model: tuned LightGBM
- Test result: `avg MAE=0.1596`, `avg R²=0.8853`

### Phase 4 — Ensemble and GNN
- Best ensemble: ridge stacking, `avg MAE=0.1543`, `avg R²=0.8912`
- Best overall model: `SchNet 3D`
- Test result: `avg MAE=0.1492`, `avg R²=0.8942`
- Gap to `R²=0.9` on the hard task: about `0.006`

Canonical summary files:

```text
results/phase4/model_comparison_final.csv
results/phase4/phase4_summary.json
TO-AI/archive/stage10_phase4_dl_ensemble_20260605.md
```

## Current model recommendation

### If you want the best accuracy
- Use `Phase 4 SchNet 3D`
- Script: `scripts/phase4/gnn_schnet_3d.py`

### If you want the best traditional / CPU-friendly model
- Use `Phase 3 tuned LightGBM`
- Script: `scripts/phase3/select_and_optimize.py`
- This is the best non-GNN result on the hard chemistry setting

## Canonical outputs

### Shared
```text
results/common/train_valid_test_split_indices.npz
results/master_experiment_log.csv
```

### Phase 2
```text
results/phase2/generalization/
```

### Phase 3
```text
results/phase3/
results/phase3/optimize/
```

### Phase 4
```text
results/phase4/
```

### Phase 5
```text
results/phase5/database/
```

Root-level files such as:

```text
results/test_predictions_*.csv
results/confidence/
results/gap_consistency/
results/scaffold/
results/tuning/
```

should be treated as older or transitional outputs unless you explicitly need them.

## Environment notes
- The project now has a real git repository: `.git/` exists.
- Use the project virtual environment:

```bash
.venv\Scripts\python.exe ...
```

- `requirements.txt` covers the main CPU pipeline, but some phase-4 code also depends on GPU-side packages installed separately in `.venv`:
  `torch`, `torch_geometric`, `torch-cluster`

## Active next steps
- Consolidate the final model comparison and reporting story.
- Decide what to report as the main final benchmark:
  `Phase 3 tuned LightGBM` vs `Phase 4 SchNet 3D`.
- Decide whether the final gap result should emphasize direct gap, blended gap, or both.
- Clean or archive root-level legacy result outputs if needed.

## Deferred work
- Curate a real commercial molecule list.
- Run `scripts/phase5/predict_commercial.py` on real supplier molecules.
- Build the final database artifact after the model/report side is stable.
