# Kunshan conjugated-component K3 protocol

Protocol date: 2026-09-06. Plan: `kunshan_discovery_plan.md`, K3.

## Causal question

K3 separates two effects that must not be conflated:

1. **Descriptor-only control:** attach the same deterministic 8-channel
   conjugated-system descriptor to every member atom and inject a learned
   32-to-192 residual after block 2.
2. **ComponentState candidate:** retain that exact descriptor path, then add a
   persistent 32-channel state for each conjugated component, with atom-to-
   component mean aggregation and low-rank component-to-atom exchange after
   blocks 3, 6, and 9.

The original GraphState9 is also trained fresh in K3a. K3a first compares
GraphState9 with descriptor-only. K3b later compares the retained descriptor-
only control with ComponentState under the same source, cache, initialization,
shuffle and training contract. This staging keeps each allocation below 12
hours while preserving causal attribution.

## Deterministic representation

Components are connected components of real bonds whose OGB conjugation flag
is true. Nonmembers use local component id `-1`. Components are ordered by
their smallest atom index. Each member atom receives the same invariant vector:
log atom count, log conjugated-bond count, cycle-rank fraction, aromatic-atom
fraction, hetero-atom fraction, branching fraction, endpoint fraction, and
aromatic-bond fraction. No target, conformer-derived distance, official
validation information, or learned preprocessing enters this cache.

The cache builder writes one atomic shard per accepted geometry shard and a
hashed manifest. CPU acceptance recomputes every component id and descriptor,
checks all 110,000 train/internal-validation graphs, and confirms sealed roles.

## Frozen training and gates

Use the accepted PCQM official-train-derived 100K/10K split. Seed42, FP32,
batch48, workers0, AdamW `1.6e-4`, weight decay `1e-6`, train-standardized L1,
cosine decay to `1e-6`, at most 40 epochs, patience8, and direct Gap supervision
remain unchanged. All new returns are bias-free and zero initialized; new
component updates contain no dropout, preserving the parent dropout draw order.

Promotion requires at least `0.001 eV` lower paired MAE, no more than 1.5x
control epoch time, and at least 15% device-memory reserve. K3b is required for
an architecture claim even if descriptor-only K3a is neutral or negative.
Official validation/test-dev, seeds43/44 and full-data training remain locked.
