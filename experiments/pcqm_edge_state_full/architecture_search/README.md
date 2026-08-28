# Official-Train Architecture Search

This bounded Track B screen asks whether architecture-only changes improve the
OGB-rich EdgeState model before any additional official full-data run is
considered. Round 1 capacity changes are closed in `round1_decision.md`, the
field-aware stem is closed in `round2_decision.md`, and the final targeted
screen is closed in `round3_decision.md`.

The separately frozen equal-ensemble confirmation passed against one control
but lost to the same-cost three-control-seed ensemble. The final attribution is
in `round4_matched_cost_decision.md`; radical context does not authorize
full-data training or official validation access.

## Fixed protocol

- Input: `nothingnessvoid/molgap-pcqm-feature-screen-20260826`, acceptance
  SHA256 `faecf13321e373e76216e7cd6a6ab64e826d6983d2e595f419874e410d0bb3a4`.
- Rows: the accepted fixed `100,000` train and `10,000` development rows, all
  drawn from the official PCQM4Mv2 training split.
- Forbidden: official validation, official test, PubChemQC labels, pretraining,
  warm starts, fusion, and production registry changes.
- Shared training: Gap-only L1, OGB categorical atom/bond features, RWSE16,
  batch 256, 20 epochs, two warmup epochs, cosine decay, seed 42.
- Control: EdgeState GPS9-192, four heads, 64 edge-state channels; seed-42
  development MAE `0.150062 eV`, radical MAE `0.253604 eV`, and non-radical
  MAE `0.141497 eV`.

## Candidates

| ID | Changed operation | Fixed configuration |
|---|---|---|
| `deep160x11` | More message-passing/global-attention depth at bounded width | GPS11-160, four heads, 64 edge-state channels |
| `edge96` | Larger persistent edge channel | GPS9-192, four heads, 96 edge-state channels |

Both Round 1 candidates were rejected at seed 42. They must not be rerun or
extended to additional seeds.

The seed-42 promotion gate requires at least `0.002 eV` overall improvement,
radical regression no worse than `0.002 eV`, and non-radical regression no
worse than `0.002 eV`. Only a passing candidate may be confirmed on seeds 43
and 44. A confirmation requires the overall direction to agree in all three
seeds. This screen never authorizes another full-data run by itself.
