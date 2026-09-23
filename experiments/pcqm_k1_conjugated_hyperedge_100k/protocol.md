# Frozen screen: K1 conjugated-component hyperedge communication

## Evidence basis

The RML corpus contains 23 trajectories and one canonical replay comparison
group at planning time. The fixed K1-v4 internal-development MAE is
0.1413736343 eV. PairToken improved 100K by 0.0030435771 eV but lost its
matched 500K advantage (candidate +0.0006985342 eV), so a further generic
atom-pair token is not favored. Persistent sparse triplets, SPD-conditioned
PairToken, function-group type tokens, slot refinements, and GPS++ local
adapters were below gate or negative. K1's own low-rank global exchange
retained its 100K-to-500K relative advantage.

In a separate GraphState family, conjugated-system ComponentState beat its
descriptor-only paired control by 0.0010280502 eV at 100K. That is not a K1
result and the GraphState full-scale route later failed. It does, however,
motivate a distinct test: can *chemically bounded* long-range exchange improve
K1 without global all-pairs capacity? [Molecular Hypergraph Neural Networks](https://arxiv.org/abs/2312.13136)
proposes conjugated structures as hyperedges; its published gains do not imply
this transplant will work.

## Single information-flow mechanism, two isolated arms

All atoms in one connected component of OGB's **conjugated-bond flag** form
one hyperedge. Only those atoms exchange a 32-channel state; atoms outside
conjugated components keep K1 unchanged. One-shot applies the exchange once
at layer 6. Persistent carries the same state through layers 3, 6 and 9.
The final component-to-atom projection is zero-initialized, making both arms
bitwise K1 at initialization. The difference isolates persistence, not width,
features, training settings, or a second unrelated module. No ETKDG coordinates,
teacher, target residual, prediction ensemble, fragment vocabulary, or
all-pairs attention enters the model.

## Preparation and identity

An independent CPU Kaggle job derives only component IDs and counts from the
accepted fixed PCQM-100K OGB graph features. Its 5,000-graph chunks have
source-index/order and SHA-256 validation, an aggregate digest, zero failures,
and explicit `gap_labels_read=false` / protected roles untouched. The CPU
sidecar is accepted and published immutably before any T4x2 training release.
The model's raw feature identity is the same nine atom categories, three real
bond categories, RWSE16 and Gap target. The sidecar is a deterministic
architecture-internal representation of those same allowed categories.

## V4/V5 screen and scale gate

Both arms use Kaggle2, fixed train 100,000 and internal development 50,000,
seed42, deterministic FP32/no TF32, physical BS128 per T4, drop-last, no
accumulation, 40 epochs/31,240 steps/3,998,720 presentations, AdamW 4e-4,
WD 1e-5, clip1, cosine to 1e-6. The frozen K1-v4 reference is not retrained.
Official validation and test-dev/challenge remain sealed. Each worker has a
separate GPU, model, optimizer, RNG and atomic output/recovery chunks.

Prospective promotion needs the frozen 0.003 eV material gain and a favorable
paired interval versus immutable K1. A positive 100K result is a shortlist,
not proof of 500K transfer. Any later 500K test requires independent budget,
matched rows/exposure/precision/batch/features, and direct paired comparison
with K1 at the same scale. No automatic multi-seed, full run, or desktop handoff.
