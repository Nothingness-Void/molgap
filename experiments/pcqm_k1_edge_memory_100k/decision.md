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
