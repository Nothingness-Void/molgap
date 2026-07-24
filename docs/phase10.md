# Phase 10: Inference, UQ, and Property Database

> This document defines the delivery contract. Live model selection belongs in
> `CURRENT_STATE.md`; task order belongs in `ROADMAP.md`.

## Goal

Build a versioned batch pipeline that accepts SMILES and emits B3LYP,
Delta-corrected near-GW HOMO/LUMO/Gap values, uncertainty, applicability-domain
signals, and provenance.

## Output Contract

Each accepted molecule must include:

- normalized input identity and validity status;
- B3LYP `homo`, `lumo`, and `gap` in eV;
- Delta-corrected near-GW values;
- calibrated target-wise uncertainty;
- continuous OOD score and trust tier;
- model/data version and failure reason when prediction is unavailable.

## Components

| Component | Implementation or evidence |
|---|---|
| B3LYP batch prediction | `src/molgap/inference.py` |
| Delta candidates | `docs/phase9.md`, `scripts/phase9/` |
| Ensemble calibration | `scripts/phase10/train_ensemble.py` |
| Embedding-distance OOD | `scripts/phase10/ood_score.py` |
| Historical v1 UQ record | `results/phase10/` |
| Historical v3 LightGBM UQ record | `results/phase10_v3/` |
| Historical v3 LoRA UQ record | `results/phase10_lora_v3/` |

Historical bundles are comparison evidence, not implicit defaults.

## Build Gates

1. Freeze the Phase 8 B3LYP base.
2. Select the Phase 9 Delta path on one fixed scaffold split.
3. Refit calibration and OOD references against that exact bundle.
4. Validate batch failure handling and all runtime constraints in `AGENTS.md`.
5. Build the commercial-molecule database only after the prediction schema and
   trust tiers are versioned.

The ordered implementation queue is in `ROADMAP.md`.

## Archive

The detailed v1 M1-UQ narrative is preserved at
`docs/archive/phase10_m1_uq_v1_history.md`.
