# Release audit — 2026-09-15

The user authorized three additional bounded rounds after the previous two
GPTrans rounds closed. The first release tested K1 real-bond storage/read
separation, not a claimed application of GPTrans PreNorm to an unnormalized K1.
K1's `_PersistentEdgeUpdate` already normalized its MLP context and residual
output; another read LayerNorm alone was rejected as a redundant hypothesis.

The GPTrans [PreNorm result](../pcqm_gptrans_relation_flow/decision.md) supported
investigating normalization placement, narrowly, without proving activation
explosion. Its [memory readback failures](../pcqm_gptrans_memory_readback/decision.md)
warned that more relation history did not automatically improve generalization.
The K1 [combined simplification](../pcqm_k1_combined_simplification_100k/decision.md)
was favorable but below the material gate; stacking was not justified.

The historical [gated-retention protocol](../pcqm_gap_architecture/local_statistics_seed42_protocol.md)
kept the original normalized edge update intact and added a rank32 gate plus
zero-initialized correction. Its [negative decision](../pcqm_gap_architecture/results/edge_retention_graphstate_seed42/decision.md)
closed that added-parameter gate. It did not test moving the existing output
normalization from persistent storage to the read view. The new arms had no
gate, extra state channel, parameter, geometry or node-to-pair path.

Evidence did not guarantee an improvement: raw persistent storage could itself
hurt bounded-horizon optimization. The two-arm [protocol](protocol.md) separated
raw versus normalized update-context reads and retained the original K1 global
path. It required identical initial tensors, not identical initial functions;
the latter would incorrectly reject a genuine immediate information-flow change.

The release carried [three-round authority](results/authorization.json).
It did not release a scale-up, extra seed or official-role use. No training
result was claimed at release; operational provenance belongs to `STATUS.md`.

## Terminal decision — 2026-09-15

Both arms completed 40 epochs and passed frozen no-inference acceptance. The
read-only-normalized arm scored `0.1417749524 eV`, regressing by
`0.0004013181 eV`. The context-and-read-normalized arm scored
`0.1404833347 eV`, improving by `0.0008902997 eV`; its paired bootstrap 95%
interval was barely favorable (`[-0.00177465, -0.00000570] eV`). Neither met
the `0.003 eV` material/run-variation gate, so `selected_candidate=null`.

The stored raw edge RMS rose smoothly from about `0.37` to `0.66–0.68` across
nine layers while every read view stayed near unit RMS. This rules out a
catastrophic activation explosion in the train-only preflight. The contrast
between arms indicates that normalizing the historical edge state before the
next update is directionally preferable to exposing the update MLP to its raw
scale; it does not establish a standalone promotion.

The isolated storage/read question is closed without another seed, norm
variant, optimizer rescue or scale-up. Because the favorable arm changes
real-bond recurrence while the earlier no-slot-attention and uniform-return
signals change the separate global slot path, round 2 may test the two pairwise
interactions independently. This is an additivity test, not automatic stacking:
third-round triple composition is conditional on accepted pairwise evidence.
Exact mechanical evidence: [acceptance](results/acceptance.json) and
[summary](results/summary.json).
