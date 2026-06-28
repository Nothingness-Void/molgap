# Phase 8 — Head-swap probes on the 500k expansion embeddings

Date: 2026-06-28

## Question

The 300k→500k expansion lifted the single-head baseline a lot. Does the larger
data change the earlier **head-swap** verdicts (MoE tie, layer-fusion mixed),
both of which were only established at 30k scale? In particular, MoE's experts
might "diverge" with more data than 30k could feed.

## Setup

All three heads trained/evaluated on the **same** expansion500k aligned
embeddings (`n_aligned = 497578`, random 80/10/10 via RandomState(42)), warm
encoders from the v3 expansion checkpoints (`phase8_gps_expansion_500k.pt`,
`phase8_schnet_expansion_500k.pt`). Internal test set, same split, so the three
rows are directly comparable.

- single head: existing `fusion_expansion_500k_metrics.json`
- MoE(4): `moe_expansion_500k_metrics.json`
- layer fusion (GPS/SchNet layers 2/4/-1): `layer_fusion_expansion_500k_metrics.json`

## Result (internal test set, 500k)

| Head | avg MAE | Gap MAE | HOMO MAE | LUMO MAE | params | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| single FusionHead | 0.08632 | 0.10043 | 0.07876 | 0.07979 | 203,907 | 14 |
| MoE (4 experts) | 0.08629 | 0.10039 | 0.07928 | 0.07920 | 409,360 | 13 |
| layer fusion (2/4/-1) | 0.08631 | 0.10002 | 0.07955 | 0.07937 | 351,363 | 9 |

Deltas vs single head:

- MoE: avg **-0.00003**, Gap **-0.00004** (params ×2.0)
- layer fusion: avg **-0.00001**, Gap **-0.00041** (params ×1.7)

## Conclusion

**Scaling to 500k did not change the head-swap verdicts — both still tie.**

- MoE's marginal gain *shrank* from 30k (≤0.0006 eV) to 500k (≤0.00004 eV),
  the opposite of the "experts need more data to diverge" hypothesis. With 2×
  the parameters it buys nothing.
- Layer fusion is also tie-level on `avg`; its only positive is a -0.00041 eV
  Gap nudge — below any practical decision threshold and not worth the 1–2 h
  layer-extraction cost over 500k molecules at inference time.

This is consistent with the standing diagnosis: the limiting factor is the
**B3LYP label ceiling**, not model capacity or fusion topology. The win from
Phase 8 came from **data coverage + trainable encoders**, not from swapping the
head. Production stays on the **single FusionHead**.

Note: these are internal-test numbers (isolates the head on identical data).
No common-eval rerun was done because the avg-MAE gaps are an order of magnitude
below the v2→v3 data-coverage gains already recorded in
`full_expansion_500k_summary.md`.

## Artifacts

- `results/phase8/moe_expansion_500k_metrics.json` / `models/phase8_hybrid_moe_e4_expansion_500k.pt`
- `results/phase8/layer_fusion_expansion_500k_metrics.json` / `models/phase8_layer_fusion_expansion_500k.pt`
- `results/phase8/fusion_expansion_500k_metrics.json` (single-head baseline, pre-existing)
