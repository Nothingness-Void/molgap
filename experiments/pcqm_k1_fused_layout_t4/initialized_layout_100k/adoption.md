# User-directed engineering selection

2026-10-12 JST. After [terminal inspection](release_002/terminal_acceptance/decision.md)
and [detailed review](release_002/terminal_acceptance/detailed_review.md), the user
requested: "Adopt the latest optimized version." Select initialized-layout,
not fused AdamW, for new compatible K1 execution declarations.

The frozen100K speed gate failed; this dated selection does not amend that
result. It is an explicit engineering tradeoff based on measured7.31% TRAIN
time reduction and byte-equal predictions/model/optimizer/RNG under one native
T4 pair. No new architecture, precision, optimizer, model promotion or
500K/full/protected-role/compute release is granted.

Reuse the registered addon and native all-arm qualification; retain the explicit
original reference and exact resume gates. Existing frozen source, plans and
jobs are unchanged. Supported default selection for new declarations is owned
by [K1 execution instructions](https://github.com/Nothingness-Void/molgap/blob/molgap-desktop/docs/operations/K1_EXECUTION.md)
after desktop integration. Formal terminal RML custody/closure and whole-question
Git routing remain pending on this owner; do not label the run replay-ready.
Selective reusable implementation integration is not a wholesale branch merge.
