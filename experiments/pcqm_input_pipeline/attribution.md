# Input pipeline attribution - 2026-10-13

## What the test supports

Fixed indexing is already implemented; no duplicate cache was added. Block
locality does not materially change the measured100K CPU traversal. The loaded
packed cache and lazy graph objects make repeated input access inexpensive here.
Adding a worker has startup/IPC costs on this Windows CPU-only traversal; that
does not establish workers0 as a faster GPU or remote default.

Size buckets genuinely reduce padded node/attention slots, but do not produce a
stable end-to-end speed benefit. K1 has only three single-slot mixers rather than
nine dense atom-atom attention blocks. Much of either model's local/edge-state
work remains on actual nodes and bonds, which buckets do not remove. This is a
mechanistic interpretation, not an exclusive operator-time attribution.

Bucket peak allocated memory is higher in both observed families. Homogeneous
large-molecule batches and shape-dependent allocation are possible explanations;
their individual contributions were not isolated. Less padding is not a proof
of lower training memory.

## Remaining discriminators

No validation labels, MAE, convergence, trained weight selection, real100K GPU
epoch, checkpoint/publication overhead, cold-storage throughput or T4/A100 cost
was measured. The timing range/direction varies across short repeated windows.
The smallest K1 block gain cannot justify changing BN/gradient trajectories
without separate quality authority and a more consequential decision question.

Sampler unit tests verify private RNG, exact membership/global tail, pass-varying
bucket companions and resumed batch order. They do not qualify complete
optimizer/checkpoint/RNG replay for a new registered training sampling policy.
Therefore training replay readiness and precision equivalence remain false.

This closes the bounded question without training a candidate endpoint. Do not
restart it or silently alter currently frozen remote sampling contracts.
