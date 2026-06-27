# Phase 10: Inference Pipeline & Property Database

## Goal
Build the user-facing prediction layer and the commercial-molecule property
database: SMILES in, B3LYP surrogate + GW correction + trust signals out.

Phase 10 absorbs the old chemical-space-screening tasks from the previous Phase 8:
element/MW/topology gates, embedding-distance OOD scoring, and commercial
molecule curation are delivery-layer concerns. They should be finalized only
after the Phase 8 production base selection.

## Current implemented subset: M1 UQ (v1-based, done)
The current UQ bundle wraps the Phase 7 hybrid + Phase 9 LightGBM Δ stack. It is
useful and validated, but it is tied to v1 embeddings. Phase 8 has now selected
`phase8_replacement_hybrid`, so this bundle remains valid only as a v1 record
until re-fit on v2 embeddings.

### 1. Δ-ensemble -> sigma
`scripts/phase10/train_ensemble.py` trains a 10-member LightGBM Δ-ensemble on the
frozen 384-d Phase 7 hybrid embedding (GPS 192 + SchNet 192). Sigma is ensemble
spread, then sigma-scaled on a scaffold-disjoint calibration set.

Test MAE is unchanged vs the single Δ model:

| target | MAE | R2 |
|---|---:|---:|
| HOMO | 0.199 | 0.85 |
| LUMO | 0.219 | 0.87 |
| Gap | 0.307 | 0.88 |

Calibration after scaling:

| target | ENCE post | 1 sigma cov | 2 sigma cov | sigma mean |
|---|---:|---:|---:|---:|
| HOMO | 0.22 | 0.72 | 0.94 | 0.26 |
| LUMO | 0.23 | 0.73 | 0.93 | 0.29 |
| Gap | 0.14 | 0.74 | 0.94 | 0.43 |

### 2. Embedding-distance OOD
`scripts/phase10/ood_score.py` computes Euclidean k-NN distance (`k=5`) to the
training embeddings.

- Distance monotonically predicts error: Gap binned MAE rises 0.239 -> 0.586 eV
  across deciles.
- Spearman rho(distance, sigma) is about 0.43-0.45.
- Cosine distance carries no signal; use Euclidean.

### 3. Single-molecule API
`src/molgap/inference.py` exposes `predict_smiles_with_uq(smiles)`:

- per-target GW estimate;
- calibrated sigma;
- underlying B3LYP prediction;
- molecule-level OOD flag.

This API currently assumes `phase7_hybrid` embeddings. Feeding v2 embeddings into
the v1 ensemble would mis-calibrate both Δ and sigma.

## Remaining Phase 10 work
After Phase 9 re-validates the GW correction on v2:

| Task | Output |
|---|---|
| P10.1 hybrid batch-predict library function | **done for B3LYP**: `load_hybrid` + `predict_smiles_batch_hybrid`; still needs near-GW wrapper |
| P10.2 batch CLI | SMILES list -> B3LYP + Δ/GW + sigma/OOD CSV |
| P10.3 in-distribution screen | element + MW + topology gates |
| P10.4 embedding OOD score | nearest-neighbor trust score for each row |
| P10.5 real-capability sounding | HOPV/full experimental comparison, layered |
| P10.6 commercial molecule universe | TCI / Sigma-Aldrich / Ossila / etc. |
| P10.7 property database | versioned near-GW HOMO/LUMO/Gap CSV with confidence |

## Artifacts
Current v1 UQ artifacts live in `results/phase10/`:

- `ensemble_calibration.json`
- `ensemble_lgbm/{target}_m0..m9.txt`
- `ood_reference.npz`
- `uq_ensemble_metrics.json`
- `ood_metrics.json`
- reliability plots and OOD plots
