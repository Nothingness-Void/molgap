# PCQM GINE Expert Continuation V4 Decision

Version 4 resumed the accepted version 3 epoch-29 optimizer, scheduler, scaler,
and model state. It reused all 11 validated graph shards and trained epochs
30-49 without rebuilding graphs.

The best checkpoint is epoch 48. Gap MAE is 0.199947 eV on the
scaffold-disjoint development split and 0.196598 eV on the fixed official-valid
5K. The official-valid result improves routed v4 by 0.095092 eV and passes the
predeclared 0.20 eV scale gate by 0.003402 eV.

All declared artifact and graph-shard SHA256 values match. The cache contains
254,997 unique, finite-label graphs with train/development/official-validation
counts of 229,335, 20,662, and 5,000. Best and last checkpoints load and agree
on best epoch 48. The prediction CSV contains 5,000 unique official-validation
indices and reproduces the reported MAE within decimal serialization error.

Decision: accept this checkpoint as the task-level PCQM Gap specialist
prerequisite for the planned Oracle-only hierarchical routing study. This does
not authorize a learned molecular Router, GPS9/fusion expansion, production
registry change, or official-test/sealed-20K access.

Machine-readable acceptance: `continuation_v4_acceptance.json`.
