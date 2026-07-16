# Roadmap — Priorities & Backlog

Tasks, milestones, and priorities only. For results/conclusions see
`CURRENT_STATE.md` and `docs/phaseN.md`. For "what's true now" see `CURRENT_STATE.md`.

## Goal
A **property database of commercially available organic molecules** — a CSV of
HOMO/LUMO/Gap at high (GW-level, **gas-phase**) accuracy. NOT limited to OLED — OLED
is one slice of the commercial-molecule set. Built on two layers:
1. the Phase 8 routed dual-GPS model — the selected B3LYP accuracy predictor (v4);
2. a **Δ-learning correction toward GW**, trained on OE62 GW5000, to push predictions
   past the B3LYP method ceiling.

The database is the deliverable; the predictor is how we build it. Delivery target:
end of Phase 11.

## Status snapshot
B3LYP-surrogate inference now has a selected **v4 routed dual-GPS accuracy
predictor** on the unchanged expansion500k data. The single-hybrid v3 remains
the component/compatibility loader. Still in **model-optimization** mode — NOT
building the database yet. Live work is re-validating Phase 9/10 GW Delta/UQ
assets against routed v4 outputs before shipping database tooling. Horizon:
**~6 months (2026 H2)**.

## Phase plan (8 → 11)

Phase 7 is a faithful B3LYP surrogate inside its training distribution. We are
**still optimizing the model**, not yet building the database. The order reflects
that: scale/architecture first (Phase 8), then lift toward GW (Phase 9), then ship
the DB (Phase 10–11).

| Phase | Theme | Question | Exit artifact |
|-------|-------|----------|---------------|
| **Phase 8** | Scaling & architecture | *Does broader data + fixed-data architecture beat v1?* | **done** — routed dual-GPS v4 selected |
| **Phase 9** | Δ-learning to GW (**next**) | *Can Delta lift routed-v4 B3LYP to GW accuracy?* | Recomputed Delta model + validation/UQ against v4 |
| **Phase 10** | Inference pipeline & property database | *Predict any organic molecule at near-GW accuracy, with trust tiers.* | Batch CLI (B3LYP + Δ → near-GW) + in-distribution screen + predicted-property DB |
| **Phase 11** | Delivery | *Ship it.* | Versioned predictor + DB, queryable access, reproducible build, data card |

The Phase 7 300k hybrid is the **v1 fallback** and stays frozen as a reference.
Phase 8 selected replacement300k v2, expansion500k v3, then the fixed-data
`phase8_routed_dualgps_hybrid` v4 accuracy path. Delta learning and UQ now get
re-validated against routed v4 predictions.

Why GW (not OLED-solid experiment): the target is *general* electronic structure,
not solid-state OLED values — so GW gas-phase quasiparticle energies are the right
high-accuracy reference, and OE62 GW5000 supplies enough clean training pairs.

### Phase 8 — Scaling & architecture (done)
Goal: push B3LYP-prediction accuracy past the 300k v1 by **expanding coverage** on
a trainable encoder. Frozen-encoder probes (MoE-on-frozen-embeddings,
descriptor-aware fusion) and the 30k trainable-encoder MoE A/B both tie the single
head, so MoE is not the default full run. The remaining lever is data coverage +
trainable encoders, validated on common OOD/hard evaluation.

| Task | ID | Status | Notes |
|------|----|--------|-------|
| Quantify coverage gaps in 300k training set | P8.1 | **done** | `results/phase8/training_space.json`; gaps = high-conjugation, narrow-gap, low S/Cl |
| Define a broader-coverage sampling spec | P8.2 | **done** | `results/phase8/sampling_spec.md`; targeted 200k top-up buckets; fetcher smoke/probe done |
| Fetch targeted replacement candidates | P8.2b | **done** | First cut uses 38,620 targeted hard rows; interrupted 200k top-up is diagnostic only |
| Assemble fixed-size replacement 300k | P8.2c | **done** | `data/raw/phase8_replacement_300k.csv`; old300k - 38,620 easy/common + 38,620 targeted hard |
| 30k trainable-encoder MoE A/B | P8.3 | **done** | MoE gain ≤0.0006 eV avg MAE; tie-level. See `results/phase8/moe_ab_30k_summary.json` |
| Common-eval old30k vs replacement30k | P8.4 | **done** | OOD-1000 neutral, P8 hard slice positive; see `results/phase8/common_eval_30k_summary.md` |
| Intermediate-layer fusion pilot | P8.4b | **done** | Internal replacement30k gain, common eval mixed; keep as head-only follow-up after full embeddings. See `results/phase8/intermediate_layer_fusion_comparison.md` |
| Build full broader-coverage graph cache (2D + 3D ETKDG, sharded) | P8.5 | **done** | 300,000 2D graphs; 298,957 3D ETKDG graphs. See `results/phase8/full_replacement_300k_summary.md` |
| Retrain full hybrid with **trainable** encoder | P8.6 | **done** | Warm-start GPS/SchNet + standard single FusionHead; common eval strongly beats P7 full baseline. See `results/phase8/full_replacement_300k_summary.md` |
| Select v2 production base | P8.7 | **done** | `phase8_replacement_hybrid` selected; see `results/phase8/v2_selection_decision.md` |
| Expand from 300k to 500k with replay + broader in-domain top-up | P8.8 | **done** | `phase8_expansion_hybrid` v3 candidate; common eval beats replacement300k. See `results/phase8/full_expansion_500k_summary.md` |
| Re-test head swaps (MoE, layer fusion) on 500k embeddings | P8.9 | **done** | Both still tie single head (MoE avg -0.00003, layer fusion avg -0.00001 eV). Head-swap route closed; bottleneck is the B3LYP label ceiling, not capacity. See `results/phase8/head_swap_500k_comparison.md` |
| Fixed-data GPS architecture + routed dual-GPS | P8.12 | **done** | v4 route improves internal/common/OOD; PCQM ties; no measurable speed penalty. See `results/phase8/gps_arch_routed_decision.md` |
| Learned utility Router for dual-GPS | AR-01 | **archived, negative** | Failed external transfer; keep fixed v4. See `results/phase8/archive/archive-r01-learned-router/decision.md` |
| Independent PubChemQC Router dataset + Late Blend | AR-02 | **archived, negative** | Oracle headroom is large, but pre-Expert routing fails. Late Blend gains `0.000881` eV Gap, below its `0.001` gate; sealed sets stayed unopened. |
| From-scratch three-expert GINE/GPS9/SchNet MoE | AR-03 | **archived, negative** | Three-expert Router fails and Geometry is removed. |
| Local GINE/GPS static blend | P8-C1 | **candidate** | Three-seed static weights pass the internal gate (`+0.001303/+0.012029/+0.003400` eV); external transfer is next. This is not a production version. |

Frozen-encoder MoE / descriptor-fusion records (done): `docs/experiment_moe_experts_2026-06-24.md`.

### Phase 9 — Δ-learning to GW (conditional on data)
| Task | ID | Notes |
|------|----|-------|
| Probe OE62 GW5000 ∩ training distribution | P9.1 | **done** — 3756 in-dist clean pairs (`results/phase9/oe62_indist.json`) |
| Recompute Δ labels for v4 | P9.2 | **next** — Δ = GW(OE62) − routed-v4 B3LYP; preserve v3 embeddings and add routed prediction/flag candidates |
| Feature study | P9.3 | OEFP vs Morgan vs GNN-embedding (molecule- AND atom-level, cf. Mezei/vL atom-resolved) as Δ-model input |
| Train Δ model | P9.4 | LightGBM / GP; scaffold split + Y-randomization; OOD molecules get Δ=const, never extrapolated. Compare external Δ vs **readout-only finetune to GW** (cf. d5sc09780k multifidelity) |
| SHAP interpretability of Δ | P9.5 | Which features drive B3LYP→GW residual? Validate real physics + report asset (ref: Dr-Islam-Lab HOMO-LUMO) |

Conditional: if the in-distribution GW subset is too small, Phase 9 degrades to a
smarter (structure-aware) bias correction rather than a full Δ model.

### Phase 10 — Inference pipeline & property database
Absorbs the old Phase 8 chemical-space-screening tasks (in-distribution screen,
embedding-distance OOD score, capability sounding): they are **delivery-layer**
trust tagging, only needed once we actually build the DB. The k-NN OOD half is
already implemented in the M1 UQ bundle.

| Task | ID | Notes |
|------|----|-------|
| Hybrid batch-predict library fn in `src/molgap/inference.py` | P10.1 | **done for B3LYP** — `load_hybrid` + `predict_smiles_batch_hybrid`; Phase 10 still needs the near-GW batch CLI |
| Batch CLI: SMILES list → B3LYP + Δ → near-GW CSV | P10.2 | Thin wrapper over P10.1 + Δ model |
| In-distribution screen (element + MW + topology gates) | P10.3 | Was P8.2. Element hard-filter ⊆ {C,H,N,O,S,F,Cl}; MW 200–1000 |
| Fingerprint / embedding nearest-neighbor OOD score | P10.4 | Was P8.3. Continuous OOD score per molecule; k-NN half done in M1 UQ |
| Real-capability sounding | P10.5 | Was P8.6. HOPV→full 127 + Hybrid + method-aligned exp comparison, layered |
| Curate commercial molecule universe (TCI / Sigma-Aldrich / Ossila / …) | P10.6 | OLED is one slice, not the whole |
| Build the property database (near-GW values + confidence/OOD flags) | P10.7 | **the deliverable** |

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
| SchNet denoising pretraining | If v2 residual analysis shows encoder pretraining is worth the cost; low priority after replacement300k win |
| LoRA / PEFT fine-tuning of encoders to GW (model-side Δ variant) | AFTER data scaling — coverage is the bottleneck first (PCQM4Mv2 coverage diagnostic). Feasible per ELoRA / GraphLoRA; GPS transformer native, SchNet linear layers adaptable. See docs/phase9.md |
| Paper figures / write-up | If advisor requires an academic deliverable |

## Done (cite, don't redo)
Phases 1-7 complete. Phase 7 = 300k scaling + GPS 2D + SchNet 3D + hybrid fusion,
OOD R² 0.941. Bootstrap CIs on the experimental comparison done
(`scripts/phase7/bootstrap_ci.py`). Training-set chemical-space characterization
done (P8.1). See `docs/phase7.md`, `docs/phase8.md`.

**A/B 3D-encoder comparison** (`scripts/ab3d/`, `results/ab3d/`): solo TensorNet
wins (Gap R² 0.906, MAE 0.222, 787k params) over SchNet (0.889, 0.239, 1.04M) and
ViSNet (0.895, 0.234, 1.10M). **But at fusion level the gap collapses to <0.2% R²**
(fusion Gap R² 0.9101 vs 0.9083), and TensorNet costs ~3.7x training time at 1M
scale for no deployment-relevant accuracy — so **production stays on SchNet**.
TensorNet remains an experimental artifact only. See `CURRENT_STATE.md`.
