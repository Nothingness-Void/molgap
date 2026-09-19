# Protocol: K1 functional-group token

## Question

Does a sparse atom-to-functional-group hierarchy improve the immutable K1-v4
100K endpoint when the hierarchy is derived only from the already accepted OGB
atom and real-bond tensors?

## Evidence basis

- PairToken is the only material 100K relation-content win in the RML, but its
  500K transfer was below the target-scale gate.
- Node-adaptive PairToken return had a favorable paired interval but gained
  only `0.0017446578 eV`, so return-allocation variants are closed.
- Historical local hierarchy supervision showed directional value, but its
  objective/schedule contract was not V5-comparable and is not reused here.
- Global motif counts, ring/path caches, edge-memory normalization, geometry,
  and graph-level fragment statistics are closed families; this candidate does
  not reopen them.

## Single mechanism

The K1 backbone, EdgeState updates, RWSE16, and global exchanges at layers
3/6/9 remain unchanged. Immediately after layer 6, current atom states are
pooled into 12 chemistry-role tokens. Each token receives only atoms bearing
that deterministic role and returns only to those member atoms. The final
64-to-192 return projection is zero initialized, so the candidate is exactly
K1 at initialization.

The role incidence is derived from existing OGB `x`, `edge_index`, and
`edge_attr`: aromatic, carbonyl, amide, ester, carboxyl, nitrile, amine,
alcohol, ether, alkene, alkyne, and ring heteroatom. No SMILES, coordinates,
new labels, protected role, or external feature is consumed by the model.

## Resource separation

1. A CPU-only Kaggle2 job builds row-aligned incidence sidecars from the
   accepted fixed cache and performs no-model acceptance.
2. The accepted aggregate SHA is frozen into the candidate source/config and
   comparison prelaunch.
3. Only then may one seed-42, one-GPU candidate run be released.

## Frozen screen

- Dataset: accepted cross-platform PCQM V4 fixed 100K/50K roles.
- Target: direct Gap only.
- Seed 42, FP32, TF32 disabled, deterministic algorithms.
- Physical batch 128, 40 epochs, 31,240 optimizer steps, 3,998,720 sample
  presentations, AdamW `4e-4`, weight decay `1e-5`, cosine schedule.
- Immutable reference: recovered K1-v4 100K V5 bundle.
- Official validation, test-dev and test-challenge remain sealed.

The materiality rule and paired uncertainty procedure will be frozen in the
post-sidecar V5 prelaunch. One discovery seed can shortlist only; it cannot
authorize seeds 43/44, 500K, full training, or protected-role access.
