# Prospective protocol — topology-defined motif graph

## Question and prior evidence

Can a non-overlapping motif graph, built only from the accepted real-bond OGB
categories, supply an information-flow level missing from K1's atom/bond
EdgeState and three single-molecule-slot exchanges? This is **not** another
MetaGIN width/depth/hop variant. The [MetaGIN2D v2 decision](../pcqm_metagin_2d_100k/attempt_v2/decision.md)
closed its independent 3-hop backbone after a large negative paired result.
The [functional-group-token decision](../pcqm_k1_functional_group_token_100k/decision.md)
found a positive but sub-gate 100K signal for sparse chemistry role content,
while repeated local capacity additions overfit. It did not create a
non-overlapping graph of motifs or exchange along its inter-motif bonds.

The [HieGT paper](https://www.icst.pku.edu.cn/huwei/docs/20241202222445429130.pdf)
is a structural prior for intra-/inter-motif communication. Its 24-layer
full-scale system also uses DFT/RDKit coordinates and different training
resources; neither its reported PCQM score nor a claim of a faithful HieGT
reproduction transfers to this pure-2D bounded study.

Alternative explanation: the existing K1 EdgeState and global slot already
encode the same bridge/motif topology; a motif graph would add capacity but
not portable information. Previous fragment, ring, path, K1-R and role-token
results keep this concern explicit.

## Authorized overnight action: CPU graph sidecar only

This action derives deterministic motif memberships and the graph of cut real
bonds from the already accepted cross-platform PCQM fixed-100K graph cache.
The three cut rules apply only to graph-theoretic bridges: ring-to-chain,
carbon-to-noncarbon in a non-ring bond, and non-single non-ring bond. All OGB
atom/bond category values remain unchanged. No SMILES reconstruction, geometry,
external row, target label, model training, or inference is involved.

The CPU job reads exactly 100,000 official-train-derived training graphs and
50,000 official-train-derived internal-development graphs, verifies the fixed
parent shard hashes and canonical source-index order, emits independently
retrievable atomic sidecar shards, and then recomputes all 150,000 partitions
for acceptance. Official validation, test-dev and test-challenge remain sealed.

Predeclared feasibility gate for considering a later GPU action:

1. Complete 150,000-row source-aligned recomputation and shard/aggregate SHA.
2. Every molecule has an exhaustive one-motif-per-atom assignment; no cut
   crosses a cyclic real bond, and every inter-motif edge maps to a cut bond.
3. At least 25% of the 100K train molecules have two or more motifs and at
   least 10% have three or more. This is a coverage screen, not a performance
   claim; failure means the mechanism is too sparse for the proposed cost.
4. No model execution and no protected-role read; CPU wall time at most two
   hours. Resource excess ends this stage as `STOP_FOR_COST`.

The job is CPU-only; it does not reserve a GPU while constructing graphs. A
valid partition does **not** by itself release training or prove a useful new
feature. The intended later scientific falsifier, if separately preflighted,
would keep K1's real-bond local state and original 100K contract, and replace
one global exchange by motif-pool → real inter-motif exchange → member-atom
return. It would compare to the immutable K1-v4 reference at seed 42, FP32,
physical batch 128, 40 epochs, with no change to data, optimizer, target or
selection. That model, its source, runtime certificate, V5 prelaunch, cost
ceiling and separate 500K NO_TRAIN portability plan must be frozen and checked
**before** any GPU submission. No GPU, second seed, 500K training, full run,
official evaluation or automatic successor is released by this CPU protocol.

The CPU sidecar is an infrastructure/feasibility record, not a training RML
replay-ready trajectory. Any subsequent training needs its own complete
prospective → trace → roles/cost → terminal RML chain; missing evidence must
not be filled by inference or retroactive guesswork.

