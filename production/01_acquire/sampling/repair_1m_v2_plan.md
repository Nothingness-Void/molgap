# Phase 8 1M-v2 Coverage Repair Plan

> **Status: closed at the pure-2D external gate.** The plan is retained as
> reproducibility context. Controlled dual-GPS evaluation regressed P8-hard and
> PCQM4Mv2-valid Gap, so the conditional 3D/fusion stage must not run. See
> `results/phase8/repair_v2_2d_external_eval/decision.md`.

## Objective

Replace the rejected 1M continuation's general-in-domain appended half without
changing the validated expansion500K base. The goal is a controlled data-mixture
test, not a new production version or a MoE experiment.

## Dataset contract

- Preserve `data/raw/phase8_expansion_500k.csv` byte-for-byte as rows 0-499,999.
- Exclude every CID and canonical SMILES in `phase8_expansion_1m.csv` when
  fetching new PubChemQC B3LYP candidates.
- Collect a 600K coverage-targeted candidate pool using
  `repair_1m_v2_sampling_spec.json`, then select at most 500K rows after
  novelty and bucket auditing. The two rarest low-Gap buckets are capped at 5K
  each after a local supply-rate pilot showed their old quotas were infeasible.
- The rejected `phase8_v4_general_topup_500k.csv` remains an evidence artifact;
  it is not included in 1M-v2 training.

## Execution boundaries

| Stage | Platform | Scope |
|---|---|---|
| Candidate fetch and selection | local machine | CSV only; no graph construction |
| 2D graph/GPS gate | SCNet | 2D GPS only |
| 3D ETKDG, SchNet, embeddings, fusion | Kaggle | only after the 2D gate passes |
| External acceptance | Kaggle | PCQM public-valid first, then common OOD/P8-hard |

## Gating

1. Candidate-pool report must show nonzero fill across every hard bucket and no
   CID/canonical-SMILES overlap with the rejected 1M table.
2. A single-head 2D GPS run with old:new replay sampling `2:1` must be
   non-regressive on PCQM public-valid and OOD before 3D construction.
3. Only a passing ordinary GPS+SchNet FusionHead may receive a head-capacity
   A/B. MoE is not part of the data-repair acceptance test.
4. A model enters the registry only after common OOD/P8-hard and PCQM checks;
   routed-v4 remains the default throughout.
