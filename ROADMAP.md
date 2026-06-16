# Roadmap — Priorities & Backlog

Tasks, milestones, and priorities only. For results/conclusions see
`CURRENT_STATE.md` and `docs/phaseN.md`. For "what's true now" see `CURRENT_STATE.md`.

## Goal
A **property database of commercially available organic molecules** — a CSV of
HOMO/LUMO/Gap at high (GW-level, **gas-phase**) accuracy. NOT limited to OLED — OLED
is one slice of the commercial-molecule set. Built on two layers:
1. the Phase 7 hybrid model — a fast B3LYP surrogate (done, OOD R² 0.94);
2. a **Δ-learning correction toward GW**, trained on OE62 GW5000, to push predictions
   past the B3LYP method ceiling.

The database is the deliverable; the predictor is how we build it. Delivery target:
end of Phase 11.

## Status snapshot
B3LYP-surrogate model is **done** (Phase 7). Live work is **Δ-learning to GW**.
No open blocker. Horizon: **~6 months (2026 H2)**.

## Phase plan (8 → 11)

The model is a faithful B3LYP surrogate inside its training distribution; the value
now is (a) knowing where it can be trusted, and (b) lifting it past B3LYP toward GW.

| Phase | Theme | Question | Exit artifact |
|-------|-------|----------|---------------|
| **Phase 8** | Chemical-space mapping & real-capability sounding | *Where is the model trustworthy, vs which reference?* | In-distribution screen + a sounding report (model vs experiment/GW, layered) |
| **Phase 9** | Δ-learning to GW | *Can a small Δ model lift B3LYP→GW accuracy?* | Trained Δ model + validation (scaffold split, Y-rand) |
| **Phase 10** | Inference pipeline & property database | *Predict any organic molecule at near-GW accuracy.* | Batch CLI (B3LYP + Δ → near-GW) + a predicted-property database |
| **Phase 11** | Delivery | *Ship it.* | Versioned predictor + DB, queryable access, reproducible build, data card |

Why GW (not OLED-solid experiment): the target is *general* electronic structure,
not solid-state OLED values — so GW gas-phase quasiparticle energies are the right
high-accuracy reference, and OE62 GW5000 supplies enough clean training pairs.

### Phase 8 — Chemical-space mapping & sounding
| Task | ID | Status | Notes |
|------|----|--------|-------|
| Characterize training set chemical space | P8.1 | **done** | `scripts/phase8/characterize_training_set.py` → `results/phase8/training_space.json` |
| In-distribution screen (element + MW + topology gates) | P8.2 | next | Element hard-filter ⊆ {C,H,N,O,S,F,Cl}; MW 200–1000 |
| Fingerprint / embedding nearest-neighbor OOD score | P8.3 | | Continuous OOD score per molecule |
| Real-capability sounding | P8.6 | | HOPV→full 127 + Hybrid + method-aligned exp comparison, layered (in-dist/OOD, conjugation, element) |

### Phase 9 — Δ-learning to GW (conditional on data)
| Task | ID | Notes |
|------|----|-------|
| Probe OE62 GW5000 ∩ training distribution | P9.1 | **done** — 3756 in-dist clean pairs (`results/phase9/oe62_indist.json`) |
| Compute Δ labels | P9.2 | Δ = GW(OE62) − model-predicted B3LYP (baseline self-supplied); needs hybrid batch predict (P10.1) |
| Feature study | P9.3 | OEFP vs Morgan vs GNN-embedding (molecule- AND atom-level, cf. Mezei/vL atom-resolved) as Δ-model input |
| Train Δ model | P9.4 | LightGBM / GP; scaffold split + Y-randomization; OOD molecules get Δ=const, never extrapolated. Compare external Δ vs **readout-only finetune to GW** (cf. d5sc09780k multifidelity) |
| SHAP interpretability of Δ | P9.5 | Which features drive B3LYP→GW residual? Validate real physics + report asset (ref: Dr-Islam-Lab HOMO-LUMO) |

Conditional: if the in-distribution GW subset is too small, Phase 9 degrades to a
smarter (structure-aware) bias correction rather than a full Δ model.

### Phase 10 — Inference pipeline & property database
| Task | ID | Notes |
|------|----|-------|
| Hybrid batch-predict library fn in `src/molgap/inference.py` | P10.1 | `load_hybrid` exists; no batch predict over the trio yet |
| Batch CLI: SMILES list → B3LYP + Δ → near-GW CSV | P10.2 | Thin wrapper over P10.1 + Δ model |
| Curate commercial molecule universe (TCI / Sigma-Aldrich / Ossila / …) | P10.3 | The database's molecule list; OLED is one slice, not the whole |
| Build the property database (commercial molecules, near-GW values + confidence/OOD flags) | P10.4 | **the deliverable** |

### Phase 11 — Delivery
| Task | ID | Notes |
|------|----|-------|
| Versioned predictor + DB, end-to-end regenerable | P11.1 | |
| Queryable access (Gradio/Streamlit or notebook) | P11.2 | Single-molecule + small-batch lookup |
| Reproducible build + data card | P11.3 | Pin torch/pyg; document provenance, schema, limitations |

## Backlog (conditional / nice-to-have)

| Task | Trigger |
|------|---------|
| Experimental-value Δ head (solid-state OLED) | If a specific OLED solid-state DB is later wanted — separate Δ from GW, needs experimental data |
| Extend training elements (Br / B / P / Si) | If OE62/usage shows too many useful molecules rejected for missing elements — needs refetch + retrain |
| Conformer-ensemble inference for flagged rows | If sounding (P8.6) shows floppy molecules dominate error |
| Better geometry via NNP (DPA-2/ANI-style) or conformer selection (CONFPASS) | LOW priority — Phase 7 conformer ensemble was only +2.5% R², so geometry is not the bottleneck. Revisit only if Δ residual analysis shows geometry/flexibility dominates error |
| LoRA / PEFT fine-tuning of encoders to GW (model-side Δ variant) | AFTER data scaling — coverage is the bottleneck first (PCQM4Mv2 coverage diagnostic). Feasible per ELoRA / GraphLoRA; GPS transformer native, SchNet linear layers adaptable. See docs/phase9.md |
| Paper figures / write-up | If advisor requires an academic deliverable |

## Done (cite, don't redo)
Phases 1-7 complete. Phase 7 = 300k scaling + GPS 2D + SchNet 3D + hybrid fusion,
OOD R² 0.941. Bootstrap CIs on the experimental comparison done
(`scripts/phase7/bootstrap_ci.py`). Training-set chemical-space characterization
done (P8.1). See `docs/phase7.md`, `docs/phase8.md`.

**A/B 3D-encoder comparison** (`scripts/ab3d/`, `results/ab3d/`): TensorNet wins
(solo Gap R² 0.906, MAE 0.222, 787k params) over SchNet (0.889, 0.239, 1.04M) and
ViSNet (0.895, 0.234, 1.10M). TensorNet is now the production 3D encoder.
