# Architecture Failure Attribution — 2026-09-08

This audit separated large architecture effects, sub-threshold directional
signals, scientific losses, and comparison uncertainty before another remote
submission.

## Findings

- Removing nine dense atom-attention blocks in favor of three GraphState
  exchanges was a large, three-seed improvement. Model size alone therefore did
  not explain performance; the allocation of global communication did.
- Five accepted fresh seed-42 GraphState controls ranged from `0.1296526` to
  `0.1303543 eV`, a `0.0007017 eV` span. Unpaired changes below `0.001 eV`
  overlap practical run variation.
- PNA statistics, edge retention, and SignNet-LapPE were neutral or negative.
  They largely repeated information already represented by local messages,
  RWSE16, EdgeState, WedgeState, and GraphState.
- Directed-bond memory improved its pair by only `0.0003956 eV`.
- Exact-shortest 2/3-hop mixing improved its pair by `0.0009066 eV`, the
  strongest retained weak positive. The candidate/control absolute-error
  correlation was `0.9075`, and only `50.84%` of validation rows improved.
- Hop-path gains were concentrated in the middle atom-count quartiles
  (`-0.00152` and `-0.00517 eV`) and middle wedge-count quartiles
  (`-0.00092` and `-0.00393 eV`). The smallest quartiles regressed by
  `+0.00185` and `+0.00131 eV`; low and high target quartiles also regressed.
  Uniform path averaging was therefore an information-selection problem, not
  evidence that more path channels or a larger cutoff was required.
- The first structure and geometry pretraining notebooks did not implement
  local masked molecular reconstruction, used schedule-mismatched intermediate
  scratch controls, and lacked full payload recomputation. Their losses closed
  those exact objectives, not all pretraining.

## Decision

Further arbitrary state accumulation was rejected. A compact `(2,1)-GT` base
replacement retained high information value but also restored quadratic global
attention, introduced new spectral/tokenization dependencies, and had a weaker
budget prior after full GPS lost decisively on this split.

The selected bounded experiment was sparse relative-value path attention on
GraphState9. It reuses the only directionally useful accepted path cache while
replacing uniform averaging with target-wise path selection and path-conditioned
values. The paired contract and exit gate are frozen in
`../../relative_value_graphstate_seed42_protocol.md`.

