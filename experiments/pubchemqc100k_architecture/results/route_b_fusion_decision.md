# PubChemQC 100K Bounded-Fusion Decision

## Decision

Retain the strict two-SchNet-pass Precision encoder combination for scale-up:

`GPS9 + GPS11-160 + primary SchNet + two-conformer-trained augmented SchNet`

Both SchNet models receive the same primary conformer at inference. The
augmented model is trained with two conformers but does not require a second
inference view.

## Fixed Scaffold-Disjoint Evidence

Values are three fusion-head seed means unless stated otherwise.

| Candidate | SchNet passes | Validation average | Test average | Test Gap |
|---|---:|---:|---:|---:|
| Minimal | 2 | 0.141685 | 0.141998 | 0.171588 |
| Cost | 2 | 0.138572 | 0.138825 | 0.167276 |
| Precision | 2 | **0.138117** | **0.138046** | **0.165819** |
| Precision | 3 | 0.136799 | 0.137347 | 0.165065 |

The two-pass Precision candidate improves over pure GPS11-160 by
`0.004424 eV` average MAE and `0.005221 eV` Gap MAE. It beats the two-pass
Cost candidate for each of seeds 42, 43, and 44.

The third SchNet forward improves average/Gap MAE by only
`0.000699/0.000754 eV`, below the `0.001 eV` retention threshold, so it is
rejected on cost.

This is a PubChemQC 100K architecture-transfer result, not a production-model
promotion. Full-scale training still requires a frozen repaired-2M protocol
and external common/OOD/P8-hard validation. No sealed-20K rows were accessed
and the production registry is unchanged.

This is Track C transfer evidence. The `route_b_*` filename is a stable legacy
artifact identifier, not the Track B ownership label.

Machine-readable evidence: `route_b_fusion_summary.json`.

The subsequent frozen-embedding head A/B supersedes the gated-sum head with a
bounded residual head. See `route_b_head_ab_decision.md`.
