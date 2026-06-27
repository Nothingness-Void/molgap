# Phase 8 v2 Selection Decision

Date: 2026-06-27

## Decision

Select **Phase 8 replacement300k hybrid** as the new v2 B3LYP base model:

- 2D encoder: `models/phase8_gps_replacement_300k.pt`
- 3D encoder: `models/phase8_schnet_replacement_300k.pt`
- fusion head: `models/phase8_hybrid_fusion_replacement_300k.pt`
- inference registry key: `phase8_replacement_hybrid`

Keep the Phase 7 hybrid (`phase7_hybrid`) as the frozen v1 fallback and
historical control. Do not overwrite Phase 7 artifacts.

## Evidence

### Shared common eval

Source: `results/phase8/full_replacement_common_eval_metrics.json`

| eval set | P7 avg MAE | P8 avg MAE | delta avg | P7 Gap MAE | P8 Gap MAE | delta Gap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.14529 | 0.12839 | -0.01690 | 0.17930 | 0.15610 | -0.02320 |
| OOD1000 | 0.12431 | 0.12144 | -0.00287 | 0.14881 | 0.14479 | -0.00402 |
| P8 targeted hard | 0.16671 | 0.13548 | -0.03123 | 0.21044 | 0.16765 | -0.04279 |

Interpretation: the full replacement300k model preserves the Phase 7 OOD1000
baseline and strongly improves the chemistry that P8 intentionally added.

### PCQM4Mv2 valid proxy audit

Source: `results/phase8/pcqm4mv2_proxy_p7_vs_p8_metrics.json`

This is the Phase 7-era PCQM4Mv2 valid coverage stress test, not an OGB
leaderboard submission. It uses official valid molecules, excludes the union of
P7 and P8 training SMILES, keeps CHONSFCl / MW 200-1000, and evaluates a
deterministic 3000-sample common ETKDG-valid subset.

| model | common n | Gap MAE | median abs err |
|---|---:|---:|---:|
| Phase 7 hybrid | 2,988 | 0.25444 | 0.17239 |
| replacement300k hybrid | 2,988 | 0.24645 | 0.16939 |
| delta P8 - P7 | - | -0.00798 | -0.00300 |

By nearest-neighbor similarity to the Phase 7 training set:

| P7 train sim bin | n | P7 Gap MAE | P8 Gap MAE | delta P8 - P7 |
|---|---:|---:|---:|---:|
| [0.0,0.3) | 182 | 0.52733 | 0.49857 | -0.02876 |
| [0.3,0.4) | 581 | 0.31846 | 0.29762 | -0.02084 |
| [0.4,0.5) | 945 | 0.22933 | 0.22353 | -0.00581 |
| [0.5,0.6) | 746 | 0.21355 | 0.21055 | -0.00300 |
| [0.6,1.0) | 534 | 0.19331 | 0.19560 | +0.00229 |

Interpretation: the gain is concentrated in low-similarity regions, matching the
P8.1 coverage-gap hypothesis. High-similarity chemistry is effectively tied.

### Error-mode audit

Source: `results/phase8/v2_error_mode_analysis.md`

The remaining common-eval worst cases are mostly flexible, large-conjugated,
S/Cl/F-containing, and narrow-gap molecules. The PCQM proxy remaining worst cases
are dominated by radical/open-shell SMILES. That PCQM residual is not the core
closed-shell commercial organic database target and should not drive the next
B3LYP base-model decision.

## Rejected alternatives

- Full 300k MoE: deprioritized. The 30k frozen-head A/B and true end-to-end 30k
  run showed tie-level or negative gains.
- Intermediate-layer fusion: not selected as the v2 base. It slightly lost to
  ordinary FusionHead on original P7 300k and had mixed common-eval behavior on
  replacement30k.
- Official PCQM4Mv2 leaderboard route: out of scope. That requires official
  PCQM4Mv2 train/valid/test-dev training and submission, not the MolGap
  PubChemQC 300k/replacement300k data.

## Handoff

Use `phase8_replacement_hybrid` for new B3LYP-base work. Before shipping the
final property database, re-run or re-validate all v1-dependent Phase 9/10 assets:

- GW Delta learning / LoRA / LightGBM residual models;
- calibrated uncertainty;
- k-NN OOD scoring based on hybrid embeddings;
- any batch inference CLI defaults and documentation that still point to
  `phase7_hybrid`.
