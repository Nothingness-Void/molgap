# GPTrans relation-flow screen

Does relation-state scale or a softmax-invisible logit offset limit the adapted
GPTrans-T core on fixed PCQM-100K? Read [decision](decision.md), then
[protocol](protocol.md). Runtime/submission truth is in [STATUS](STATUS.md).
Compact audit evidence: [audit](results/audit.json).

Identity note: this server experiment is **Pair PreNorm** (`pair_prenorm`),
with accepted development MAE `0.1535023336 eV` and gain `0.0031248707 eV`.
It is not the separately developed desktop Pair Update Norm mechanism, and the
server repository does not claim authority for that desktop result.
