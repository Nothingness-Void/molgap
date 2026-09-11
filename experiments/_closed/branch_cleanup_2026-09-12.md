# Branch cleanup record — 2026-09-12

The archive branch retained the complete histories below with an `ours`
strategy merge. This deliberately preserved the existing archive checkout
while making every retired tip and its exact historical tree reachable.

## Retired histories

| Former ref | Retained tip | Disposition |
|---|---|---|
| `codex/pcqm-geometry-scratch-control` | `de0f4ddb62c9e23257ec225fca1e4e9ba1272b3b` | Closed geometry warm-start and GraphState scale history; includes the former warm-start branch ancestry |
| `codex/pcqm-evidence-guided-scnet` | `dcf43059be789d72c67f72361c5f8131bd198368` | Superseded SCNet hierarchy/geometry infrastructure history |
| `codex/pcqm-local-hierarchy-scnet` | `8226fbf9590cfc8f5b60c18c14d8f8e1bcc08977` | Superseded hierarchy replication history |
| `codex/scnet-qm9-charge-adapter` | `70886953977b593e08271f8d987d65d8ef76b96c` | Seed-42 scientific rejection |
| `codex/scnet-qm9-masked-charge-pretrain` | `8ea728fc53432f2d6bb324ba519965083dbf571c` | Equal-exposure scientific rejection |
| `codex/presentation-figures-assets` | `adc4d5fcbea239b33de3ab958df7bfd96e1e5be4` | Inactive presentation assets |
| `imgbot` | `f0a816355da5cf68ef0d378b5067eb979c0fd64a` | Superseded formatting-only history |

The local `codex/pcqm-torsion-seed42` ref was removed after its tip
`005a02208e92712e69b0bc79672be964e1f07f09` was verified reachable from both
`molgap-server` and `archive`.

## Refs deliberately retained

| Ref | Tip | Reason |
|---|---|---|
| `codex/scnet-esgps6-304-500k` | `03f5253dcf86edb87512a4bbd5dcececa5a7c0dd` | Positive 500K relation-pretraining result awaiting owner integration |
| `codex/scnet-gptrans-t-500k` | `c86970d0e309f23c4380de8d5935ec961d0bb209` | Positive 500K architecture result awaiting owner integration |
| `codex/scnet-distance-angle-triangle-500k` | `a701e36860112f36385d107d0c9e3cc337b98afc` | Launch record has no terminal decision in Git |
| `codex/exp/gptrans-k1-geometry-500k` | `01d5c88cb4118e9cc6888440a5e1842ddaf12988` | Created concurrently during cleanup for an active 500K geometry-transfer screen |

The redundant `codex/scnet-edgestate304-500k` ref at
`f19bfd5df9e7b3c35aea2cf8072ea7657b8af73d` was removed only after verifying
that its complete history is an ancestor of the retained ESGPS6 branch.

The four long-lived refs remain `master`, `molgap-server`, `molgap-desktop`,
and `archive`. No commit was merged into or removed from `molgap-desktop` by
this cleanup.
