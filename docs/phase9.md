# Phase 9: Δ-learning to GW

## Goal
Lift predictions past the B3LYP method ceiling toward **GW gas-phase** accuracy,
without retraining the 300k model. The B3LYP surrogate is faithful inside its
distribution (OOD R² 0.94) but B3LYP itself systematically misjudges orbital
energies (esp. underestimates the gap). A small **Δ model** corrects this.

## Why decoupled (not retraining on corrected labels)
Add the correction at the **output**, not the label:
```
y_pred(X) = GNN(X)            # frozen B3LYP surrogate, predicts B3LYP
          + f_Δ(X)            # small model, only inside the experimental/GW domain
```
- A *value-only* linear correction (Δ as a function of the B3LYP value) is
  mathematically equivalent to post-processing the existing output — retraining
  300k would be wasted. The current constant bias-correction is exactly this.
- A *structure-dependent* Δ has real signal, but baking it into 300k pseudo-labels
  would extrapolate a narrow-domain rule onto the whole set and **poison** clean
  B3LYP labels. Decoupling avoids both: GNN untouched, Δ trained on its own data,
  OOD molecules get Δ=const (never extrapolated).

## Δ definition (baseline is the model, not true DFT)
```
Δ(X) = y_GW(X) − y_model-B3LYP(X)
```
At inference there is no real DFT run, so the baseline is the model's own output;
Δ therefore absorbs both the B3LYP→GW method gap and the model's B3LYP-fit error.

## Training data (P9.1, done)
OE62 GW5000 (G0W0@PBE0, gas-phase) ∩ our training distribution.
- Source: TUM `df_5k.json` (`data/raw/oe62_df_5k.json`), probe in
  `scripts/phase9/probe_oe62_indist.py` → `results/phase9/oe62_indist.json`.
- **5239 GW molecules → 3756 in-distribution clean pairs** (elements ⊆
  {C,H,N,O,S,F,Cl}, MW 200–1000). Far above the ~150–300 needed.
- Rejected by foreign element: Br 346, P 286, Si 169, I 65, Se 55, B 34, …
- Caveat: OE62 GW gap (median 7.93 eV) >> training B3LYP gap (median 4.81 eV) —
  GW opens the gap (that's the signal), but OE62's crystal-sourced molecules skew
  harder/wider-gap. The 3756 is an *element+MW* filter; P8.3 fingerprint/embedding
  distance can refine it further. Geometry mismatch (OE62 PBE+vdW crystal vs our
  ETKDG) is absorbed into Δ.

### Rejected data source: QM9-GW 134k (Fediai et al., Sci. Data 2023)
134k QM9 molecules with GW HOMO/LUMO + paired B3LYP — conceptually ideal and 25×
larger than OE62. **Not used:** QM9 is small molecules (≤9 heavy atoms, MW <150),
which has ~zero overlap with our training distribution (MW 200–1000). Our model
would be OOD on every QM9 molecule → dirty B3LYP baseline → poisoned Δ labels.
Size ≠ usefulness: OE62's 3756 in-distribution pairs beat QM9's 134k all-OOD pairs.
Revisit only if the project expands to small molecules (needs retrain) or for
cross-scale multifidelity pretraining (risky extrapolation).

## Results (P9.4 variant A — LightGBM learns Δ, DONE)
`scripts/phase9/train_delta.py` → `results/phase9/delta_model_metrics.json`.
Scaffold split (3111 unique scaffolds / 3736 mols → train 3041 / test 695,
scaffold-disjoint). Features = 192+192-d hybrid embedding. Test-set GW-accuracy MAE:

| target | raw | const | **Δ model** | Yrand | R²(Δ) |
|--------|-----|-------|-------------|-------|-------|
| HOMO | 2.205 | 0.283 | **0.197** | 0.283 | 0.856 |
| LUMO | 0.985 | 0.298 | **0.217** | 0.299 | 0.876 |
| Gap  | 3.159 | 0.471 | **0.303** | 0.468 | 0.885 |

All three pass: Δmodel < const (learns structure, −27..−36% over constant bias),
Yrand ≈ const (signal real, not overfit), R² 0.86–0.89. From raw B3LYP this is a
~10× error cut (Gap 3.16 → 0.30 eV). **Δ-learning works**; we have a B3LYP→GW
correction layer at ~0.24 eV mean GW accuracy. Caveats: in-dist OE62 (incl. PBE-vs-
ETKDG geometry noise), gas-phase GW (not solid-state), commercial molecules still
need OOD flagging at DB time.

Models saved: `delta_lgbm_{homo,lumo,gap}.txt` (write via Python, not LightGBM C
save_model — the latter mangles the non-ASCII "文档" path).

## Geometry-noise diagnostic (DONE — geometry is NOT the bottleneck)
`scripts/phase9/diagnose_geometry.py` rebuilds the 3D graph from OE62's own PBE
geometry (vs ETKDG) and retrains Δ. If geometry mismatch were a big noise source,
the "perfect" PBE geometry should cut error a lot. It barely does:

| target | ETKDG baseline | PBE geometry | improvement |
|--------|----------------|--------------|-------------|
| HOMO | 0.197 | 0.198 | −0.001 |
| LUMO | 0.217 | 0.198 | +0.019 |
| Gap  | 0.303 | 0.293 | +0.010 |

Only 0.01-0.02 eV gain from perfect geometry → **ETKDG-vs-PBE mismatch is a minor
noise source**. So NNP geometry / conformer ensembles are NOT worth pursuing. The
remaining 0.2-0.3 eV is the method floor (GW self-error ~0.1-0.2 + B3LYP transfer
~0.12 + small feature gap) — the Δ model is essentially at the physical limit.
Decision: accept the variant-A Δ model, proceed to Phase 10 (don't chase margins).

## Method ladder (start simple, climb with data)
| Stage | f_Δ | Features | Data |
|-------|-----|----------|------|
| 0 (done) | constant bias | — | any |
| **1 (next)** | LightGBM / GP | 192-d GNN embedding (+ select RDKit descriptors / OEFP) | 100s–1000s ✅ |
| 2 | shallow MLP head | embedding | 500+ |
| 3 (far) | end-to-end fine-tune | atom graph | thousands |

Data-efficiency comes from reusing the **GNN embedding** as features (structure
already encoded), not relearning representation. 3756 pairs comfortably support
Stage 1.

## Validation (required)
- **Scaffold split** (group by Murcko scaffold) — random split leaks similar
  scaffolds and overstates accuracy.
- **Y-randomization** — shuffle Δ labels, retrain; performance must collapse, or
  the model is fitting noise.
- **SHAP interpretability** — which features drive the B3LYP→GW residual? Confirms
  Δ learned real physics (e.g. conjugation / heteroatom effects) and is a
  report/defense asset.

## External reference (what we borrowed)
`Dr-Islam-Lab-Group/HOMO-LUMO` (GDB13, HF-level, descriptor + LightGBM/Bi-LSTM/MLP
ensemble, SHAP). Overall **below this project** in accuracy (HF vs B3LYP→GW),
target (gap-only vs HOMO/LUMO/Gap), and representation (descriptors vs GNN). Its
"HF + empirical conversion" is just a value-only linear Δ. Useful takeaways:
**(1) SHAP for Δ interpretability** (adopted into validation above);
**(2) LightGBM on descriptors works** — corroborates the Stage-1 choice. Not
adopted: multi-model ensemble (overkill / overfit risk on 3756 points).

## Literature: ML-corrected DFT (positioning)
Δ-learning correction of DFT is an established paradigm — our direction is backed,
not novel-risky. Two references and what we take / leave:
- **Mezei & von Lilienfeld, JCTC 2020** (arXiv:1903.09010) — post-hoc Δ-learning
  correcting 6 DFAs toward CCSD(T) for *noncovalent energies*, FCHL **atomic**
  representation, atom-resolved correction. **Take:** try atom-level GNN-embedding
  aggregation as a Δ feature (P9.3), not just molecule-level. **Leave:** it corrects
  energies, we correct orbital levels — different property, don't copy its loss.
- **arXiv:2504.14961 (2025)** — ML correction of B3LYP toward the *exact XC
  functional*, density-based, SCF-embedded (double-cycle), trained on absolute
  references to avoid error-cancellation → transferability. **Take:** the
  "absolute reference + diverse chemistry → transferable" rationale (we use generic
  OE62 GW absolutes, matching this). **Leave:** SCF-internal correction is far too
  heavy; our post-hoc orbital-level Δ is the right lightweight choice.

General lesson: train on absolute GW values across diverse chemistry (not a narrow
family) for transferability; consider atom-resolved features.

### Multi-fidelity transfer learning (key reference, Chem. Sci. 2026, d5sc09780k)
Most on-target paper found. GNN pretrained on DFT/TDDFT, **finetuned with limited
qsGW / qsGW-BSE** data — exactly our "abundant B3LYP + scarce GW" regime. Reports
pretraining improves accuracy, cuts reliance on costly GW data, and **mitigates
outliers even for molecules larger / chemically distinct from finetuning set**
(directly eases our OE62 distribution-shift worry).
- **Take 1 — readout-only finetuning as a baseline alongside the Δ model.** Their
  "freeze encoder, finetune readout to GW" is the neural twin of our "freeze GNN +
  external Δ head". Compare both in P9.4: external Δ (LightGBM/GP on embedding) vs
  readout-only finetune to GW.
- **Take 2 — multifidelity/pretraining helps OOD**, so OE62's harder/wider-gap
  skew is less threatening than feared.
- **Leave:** their BSE excitation energies (optical, excitonic) — out of scope; we
  only need the qsGW quasiparticle part. Note qsGW (self-consistent) vs OE62
  G0W0@PBE0 (one-shot) differ slightly in fidelity tier.

## Model-side variant: LoRA / PEFT (literature, conditional)
Beyond the frozen-encoder + LightGBM head, parameter-efficient fine-tuning (LoRA)
could let the encoders *lightly* adapt to GW with few trainable params (anti-overfit)
— a middle ground between fully frozen and full finetune. Validated for molecular
GNNs and GNN transfer in recent work:
- **ELoRA** — LoRA on SO(3)-equivariant GNNs, improves molecular energy/force → LoRA works on molecular GNNs.
- **GraphLoRA** (arXiv 2409.16670) — LoRA for cross-graph GNN transfer, ~20% params → our exact transfer setting.
- **MMEA** (arXiv 2511.06696) — PEFT adapter for equivariant GNNs.
- **PEFT review** (arXiv 2501.00365) — LoRA ≈ 95-100% of full finetune at 0.01-1% params.

Fit: GPS transformer layers are a native LoRA target; SchNet interaction/filter
linear layers can take adapters too. Caveats: ELoRA targets *equivariant* GNNs
(SchNet is invariant — idea transfers, not identical); all 2024-2025, no off-the-shelf
"B3LYP→GW via LoRA". **Priority: coverage is the bottleneck (PCQM4Mv2 coverage
diagnostic), so LoRA is a model-side refinement AFTER data scaling + retrain.**
(Raised by Omozawa in lab discussion 2026-06.)

### LoRA FusionHead pilot (2026-06-24)

Implemented a low-risk LoRA feasibility test that does **not** overwrite or alter
the B3LYP production checkpoint:

- base: Phase 7 `hybrid_fusion_optuna.pt` FusionHead initialized to B3LYP;
- frozen: all original FusionHead linear weights;
- trainable: LoRA adapters on the 6 Linear layers only;
- input: existing OE62 `emb_2d` + `emb_3d` frozen embeddings;
- target: absolute GW HOMO/LUMO/Gap;
- split: same scaffold-disjoint OE62 split as `train_delta.py` test set.

Results on scaffold-test OE62 GW (`n_test=695`):

| method | trainable params | HOMO MAE | LUMO MAE | Gap MAE | avg MAE | avg R² |
|---|---:|---:|---:|---:|---:|---:|
| raw B3LYP | 0 | 2.205 | 0.985 | 3.159 | 2.116 | -4.123 |
| const Δ | 0 | 0.283 | 0.298 | 0.471 | 0.351 | 0.762 |
| LightGBM Δ | — | 0.197 | **0.217** | **0.303** | 0.239 | — |
| LoRA r=4 | 8,460 | 0.189 | 0.222 | 0.311 | 0.241 | 0.867 |
| **LoRA r=8** | **16,920** | **0.186** | 0.218 | 0.303 | **0.236** | 0.877 |
| LoRA r=16 | 33,840 | 0.190 | 0.218 | 0.306 | 0.238 | **0.879** |

Artifacts:

- script: `scripts/phase9/train_lora_fusion_delta.py`
- checkpoints: `models/hybrid_fusion_lora_gw_r{4,8,16}.pt`
- metrics: `results/phase9/lora_fusion_delta*_metrics.json`
- predictions: `results/phase9/lora_fusion_delta*_predictions.csv`

Conclusion: **LoRA is feasible and competitive with the current LightGBM Δ
baseline even in the conservative FusionHead-only form.** Rank 8 gives the best
average MAE and beats LightGBM on HOMO, ties Gap, but is slightly worse on LUMO.
This is not yet encoder LoRA; the next meaningful step is a true adapter path on
GPS/SchNet encoder linear layers, still saved separately so the B3LYP base model
remains unchanged.

### Encoder LoRA pilot (2026-06-24)

Next, injected LoRA into the frozen Phase 7 encoders plus FusionHead, rebuilding
the OE62 2D/3D graphs from SMILES so gradients flow through `GPSWrapper.encode`
and `SchNetWrapper.encode`. Original B3LYP weights are still frozen; adapters are
saved separately. PyTorch `MultiheadAttention.out_proj` is skipped because its
forward path accesses `.weight` directly; GPS still adapts through node/edge
embeddings and GINE/MLP linear layers.

Single-seed probe: rank 4, alpha 8, scaffold-test OE62 GW (`n_test=695`).

| adapter targets | trainable params | HOMO MAE | LUMO MAE | Gap MAE | avg MAE | avg R² | note |
|---|---:|---:|---:|---:|---:|---:|---|
| FusionHead LoRA r=8 | 16,920 | 0.186 | 0.218 | 0.303 | 0.236 | 0.877 | previous best lightweight adapter |
| GPS + Fusion | 66,892 | 0.185 | 0.200 | **0.270** | 0.218 | 0.875 | strong gain; fastest encoder adapter |
| SchNet + Fusion | 71,936 | 0.186 | 0.206 | 0.280 | 0.224 | **0.895** | better R², slower, more overfit-prone |
| **GPS + SchNet + Fusion** | **130,368** | **0.182** | **0.197** | 0.272 | **0.217** | 0.890 | best MAE, but train/val gap suggests overfit risk |

Artifacts:

- graph cache: `results/phase9/delta_oe62_graphs.pt`
- script: `scripts/phase9/train_encoder_lora_delta.py`
- checkpoints: `models/hybrid_encoder_lora_gw_{gps_fusion,schnet_fusion,gps_schnet_fusion}_r4.pt`
- metrics: `results/phase9/encoder_lora_delta_*_metrics.json`

3-seed stability check (`seed={42,1,2}`, same scaffold split seed 42):

| adapter targets | trainable params | HOMO MAE | LUMO MAE | Gap MAE | avg MAE | avg R² | best-val MAE | train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPS + Fusion | 66,892 | 0.185 ± 0.004 | 0.204 ± 0.005 | 0.275 ± 0.005 | 0.221 ± 0.004 | 0.873 ± 0.016 | 0.188 ± 0.005 | 181 ± 76 s |
| **GPS + SchNet + Fusion** | **130,368** | **0.183 ± 0.001** | **0.197 ± 0.002** | **0.270 ± 0.003** | **0.217 ± 0.002** | **0.895 ± 0.005** | **0.179 ± 0.003** | 411 ± 47 s |

Conclusion: **true encoder LoRA is the strongest GW adaptation tried so far**,
beating LightGBM Δ and FusionHead-only LoRA by a meaningful margin on scaffold
test. The multi-seed check changes the recommendation: GPS+Fusion is the fast,
lower-capacity fallback, but **GPS+SchNet+Fusion is stable enough to be the
primary neural GW-adaptation path**. The remaining risk is not seed instability;
it is calibration / applicability-domain behavior on commercial OOD molecules.
Next promotion gate: add UQ/calibration and OOD guardrails before replacing the
LightGBM Δ baseline in any user-facing workflow.

## Conditional
If P8.3 refinement shrinks the clean set too far, Phase 9 degrades to a
structure-aware (but simpler) bias correction rather than a full Δ model.

## v3 revalidation (2026-07-01)

After Phase 8 promoted `phase8_expansion_hybrid` to the default B3LYP base, the
GW Δ-learning stack was re-run with v3 embeddings and v3 B3LYP predictions.

LightGBM Δ on OE62 scaffold split:

| model | feature mode | HOMO MAE | LUMO MAE | Gap MAE | Gap R² |
|---|---|---:|---:|---:|---:|
| v1 LightGBM Δ | embedding | 0.197 | 0.217 | 0.303 | 0.885 |
| v3 LightGBM Δ | embedding | 0.185 | 0.216 | 0.300 | 0.895 |
| v3 LightGBM Δ | embedding + descriptors + B3LYP pred | **0.184** | **0.212** | **0.288** | **0.904** |

Encoder LoRA was also re-run against v3 (`GPS + SchNet + Fusion`, r=4,
3 seeds). It is the highest-accuracy GW candidate so far:
HOMO/LUMO/Gap MAE = `0.184±0.003 / 0.186±0.002 / 0.260±0.006`.

Phase 10 was re-calibrated for the descriptor-enhanced v3 LightGBM Δ baseline in
`results/phase10_v3/`; load it explicitly with
`load_uq_bundle(results_subdir="phase10_v3")`.

Decision record: `results/phase9/v3_delta_decision.md`.
