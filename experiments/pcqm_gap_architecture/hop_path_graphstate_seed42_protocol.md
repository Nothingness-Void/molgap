# PCQM hop-path GraphState seed-42 protocol

Protocol date: 2026-09-07. This is the final bounded random-initialized
architecture question retained by the incremental literature audit. It tests
one information channel inspired by MetaGIN: exact-shortest two/three-hop path
statistics. It is not a MetaGIN reproduction and makes no claim that graph
hops are physical 3D coordinates.

## Hypothesis

The frozen GraphState9 encoder already contains real-bond EdgeState, sparse
wedge state, ETKDG bond distances/angles, RWSE16 and a shared molecular state.
It does not explicitly distinguish how many chemically different shortest
two/three-hop paths connect an ordered atom pair. A narrow shared path mixer
may expose conjugated and alternative local routes without restoring global
attention or enumerating torsion coordinates.

## Frozen comparison

Train a fresh
`ogb_distance_angle_triangle_edge_state_graph_state9` control beside exactly
one candidate,
`ogb_distance_angle_hop_path_triangle_edge_state_graph_state9`. Reuse the
official-train-derived 100,000 train / 10,000 internal-validation roles and
the accepted ETKDGv3+MMFF94s single-conformer geometry cache. Keep seed 42,
FP32, batch 48, AdamW `1.6e-4`, weight decay `1e-6`, normalized L1, cosine
decay to `1e-6`, 40 epochs and patience 8. Predict Gap directly. Official
validation and test-dev remain sealed.

The Kaggle job requests T4x2. GPU 0 owns the fresh control and GPU 1 owns the
candidate, with independent process, RNG, model, optimizer, checkpoint and
output directories. The internal GPU budget is 14,400 seconds. No local model
execution is permitted.

## Immutable path cache

The CPU cache is derived only from the accepted 100K/10K graph/geometry cache.
For every ordered atom pair at exact shortest distance two or three, enumerate
all simple shortest paths and store one directed relation with eight static
features:

1. two-hop indicator;
2. three-hop indicator;
3. `log1p` shortest-path multiplicity;
4. mean fraction of single bonds across the paths;
5. mean fraction of double bonds;
6. mean fraction of triple bonds;
7. mean fraction of aromatic bonds;
8. fraction of paths whose full bond sequence is conjugated.

The cache records per-role relation counts, multi-path counts, maximum path
multiplicity, shard hashes and the parent graph/wedge/geometry identities.
CPU acceptance must pass before a GPU kernel can be submitted.

## Candidate mechanism

At blocks 3, 6 and 9, one shared rank-32 mixer projects source atom, target
atom and static path features into a separately normalized sparse message.
Messages are mean-aggregated to the target atom. The return projection is zero
initialized, so every shared parameter and the candidate's initial function
match the fresh control. The baseline remains free of path inputs. Expected
parameters are 3,665,809 for the control and 3,697,537 for the candidate,
both below the 4M ceiling.

## Decision boundary

Mechanical acceptance requires exact source/cache hashes, T4x2 isolation,
finite predictions/loss/gradients, nonzero gradient on the zero-start return,
aligned 10,000-row validation payloads, atomic model/checkpoint/trace artifacts
and sealed-role flags. Scientific ranking uses only the fresh paired internal-
validation MAE, throughput and peak memory.

The candidate becomes promising only if it lowers MAE by at least `0.001 eV`,
runs at no more than 1.5x the control epoch time and retains at least 15%
device-memory reserve. A smaller positive result is recorded as weak evidence
and closes without more seeds. Completion never authorizes seed 43/44,
full-data training, official evaluation or molecular-research-server access.

Primary method reference: [MetaGIN](https://doi.org/10.1007/s11704-024-3784-y).
Local transfer judgment:
[`literature_extension_2026.md`](literature_extension_2026.md).
