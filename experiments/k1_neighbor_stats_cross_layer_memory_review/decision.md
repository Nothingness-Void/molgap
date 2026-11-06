# K1 neighbour statistics and cross-layer associative state: planning review — 2026-11-06

No training, no probe and no submission. Two pure-2D K1/EdgeState directions were reviewed
separately (not fused) by a multi-round A/B debate and an independent ruling; both close as
`NO_TRAIN` / duplicate. Neither used a protected role or any Kaggle quota.

## Direction 1: PNA-style low-order statistics of the K1 neighbour messages — STOP

K1's local operator is `ResGatedGraphConv` with `aggr="add"` over sigmoid-gated, edge-conditioned
messages. On the fixed-100K train shard 0 (50,000 graphs, 696,600 nodes) in-degree is 1: 25.0%,
2: 45.8%, 3: 27.6%, 4: 1.6% (mean 2.06). At degree 2, max plus min equals the sum, so max/min/std add
only the per-channel difference of two messages for 46% of nodes, and the mean adds nothing the
sum and the known degree lack. The same family has been tested on molecules without a retained
gain: a historical GraphState PNA adapter (2026-09-07, a different 3D-geometry base, 0.1717806 meV
worse, +58,946 parameters, -24.3% throughput), SSMA on the gated messages (+2.067 meV, +48.977% step
time), extremal pooled statistics (+3.333 meV) and degree-balance (-1.277 meV at 100K, +2.171 meV on
frozen fixed500K). The Desktop single-model review says not to reopen PNA statistics under a new
name. A frozen-feature probe would measure readability, not training benefit, and no unspent role is
allowed for it. Reopening needs an explicit user waiver of that rule and an approved hook into the
local block; the recorded design (gated layer-3 per-channel max/min/std over incoming messages,
LayerNorm, Linear(576,32)-SiLU-Linear(32,192) with a zero-initialised output, about 25K parameters,
a strict same-run pair, final-pass -3.0 meV gate, 1.5 meV stop point, no early reading) is kept here
only as that waiver's starting point.

## Direction 2: CoAtGIN cross-layer associative state — STOP

The official `AttMessage` accumulates the key-value numerator across layers but recomputes the
key-sum denominator from the current layer only, so the accumulated state is not a normalised memory
and its output scale grows with depth; it should not be called an exact memory. A normalisation-
consistent accumulation is a different mechanism, so a result for it says nothing about CoAtGIN. K1
already has graph-level, node-queried slot state at layers 3, 6 and 9 and the residual stream carries
earlier layers. The closest K1 test, current-layer linear attention, was +0.399 meV at 100K and
+2.953 meV on frozen fixed500K after an early lead; accumulated pair-state readback on the 5,246,817-
parameter GPTrans base was +1.243 and +1.594 meV worse, which closes that mechanism family but is not
a K1 result. The frozen reference has no key/value maps, so a probe cannot discriminate.

## Desktop and Kaggle1 overlap

Git shows no PNA, CoAtGIN or associative-memory plan, job or code on any of 46 remote branch tips;
Desktop's live K1 jobs are a Kaggle2 Muon 500K pair and a Kaggle3 SwiGLU prenorm 100K run, and
Kaggle1 work since 2026-11-01 is GPTrans only. Not determinable from Git: Kaggle1's live state and any
unpushed Desktop work. Before any future submission: re-fetch and re-search all remote branches, ask
the user to confirm with the Desktop owner, and list the running kernels on kaggle2 and kaggle3.
