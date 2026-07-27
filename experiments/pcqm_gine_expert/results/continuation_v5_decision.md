# PCQM GINE Expert Continuation V5 Decision

Version 5 resumed the accepted version 4 epoch-49 model, optimizer, scheduler,
and scaler state. It reused the same 11 validated graph shards and trained
epochs 50-69 without rebuilding graphs.

The best checkpoint is epoch 68. Gap MAE is `0.191690 eV` on the frozen
scaffold-disjoint development split and `0.187320 eV` on the fixed
official-validation 5K. The fixed-5K result improves version 4 by
`0.009278 eV` and routed v4 by `0.104370 eV`.

The 5,000 prediction rows reproduce the reported MAE at
`0.187320007 eV`. Best and last checkpoints load at epochs 68 and 69, and all
downloaded artifact hashes match the completion manifest.

Decision: accept version 5 as the strongest current task-level PCQM Gap
specialist. This is still a 250K-sample local protocol result, not a
PCQM4Mv2 leaderboard result. It does not authorize official-test access,
sealed-20K access, learned molecular routing, or production-registry changes.
