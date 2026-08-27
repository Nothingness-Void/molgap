# PCQM4Mv2 Gap-only 100K Architecture Screen

## Question

Can persistent real-bond EdgeState improve a matched Structural GPS9 baseline
on the official PCQM4Mv2 Gap task while retaining a credible path to one A100
run shorter than 12 hours?

## Official-data boundary

- Source: Kaggle mirror `piero0/pcqm4mv2`, with columns `idx`, `smiles`, and
  `homolumogap`.
- Only the first `3,378,606` rows, the official OGB training role, may be read.
- A deterministic seed-42 sample supplies the initial `100,000`
  development-train rows and `10,000` internal-validation rows. A disjoint
  1,024-row reserve stream is drawn from the same frozen RNG state. If OGB/RDKit
  cannot construct an initial graph, the next constructible reserve row fills
  that exact role and slot. Every failed attempt and replacement is retained in
  hashed ledgers; no failure is silently filtered.
- The official validation and test-dev roles stay unread. The cache and every
  downstream output must state both role-read flags as `false`.
- Graphs use `ogb.utils.smiles2graph`: nine categorical atom fields, three
  categorical real-bond fields, and no coordinates or external labels.
- RWSE16 is computed only from the real-bond topology.

The accepted cache owns the initial, reserve, and effective index hashes, exact
source identity, feature ranges, failure and replacement ledgers, shard hashes,
and aggregate hash. Acceptance requires exactly `100,000` effective train
graphs, `10,000` effective internal-validation graphs, and zero unresolved slots. A GPU
run may use only that immutable cache.

## Matched first round

Two models are trained sequentially in one Kaggle GPU task from random
initialization:

1. `ogb_structural_gps9`: OGB categorical encoders, RWSE16, nine 192-wide GPS
   blocks, four attention heads, mean pooling, scalar Gap head.
2. `ogb_edge_state_structural_gps9`: the same graph input, width, depth,
   attention, pooling, and scalar head, with a persistent 64-wide real-bond
   state updated at every layer.

Shared training contract: seed 42, FP32, batch 48, AdamW, learning rate
`1.6e-4`, weight decay `1e-6`, at most 40 epochs, patience 8, normalized scalar
L1 training loss, and eV MAE for selection. There is no warm start,
pretraining, target residual, prediction fusion, 3D input, or test evaluation.

The historical official-PCQM GPS search supplies the common optimizer scale;
it supplies no checkpoint. Training both models afresh on the same frozen
cache is required for an architecture claim.

## Advancement and stop rules

- Seed 42 is a feasibility gate. EdgeState advances only if its best internal
  validation Gap MAE is strictly lower than the newly trained baseline and all
  artifacts pass acceptance.
- A seed-42 winner is not final. Seeds 43 and 44 must use the same split and
  frozen contract; the candidate must improve in all three seeds and improve
  the three-seed mean.
- Selection stays on internal validation. Official validation is reserved for
  one final, frozen model decision and test-dev for final inference only.
- Failed candidates are closed as architecture results; no seed, width,
  schedule, or threshold retry is allowed under the same question.
- Only the final Kaggle-selected architecture becomes eligible for a
  train-role-only A100 throughput benchmark on the molecular-research server.
  No server access or full training occurs before that decision.

## A100 budget gate

Before any full-data training, the frozen winner must demonstrate at least
`1,800 graphs/s`, no measured epoch above 32 minutes, projected end-to-end
training no longer than 10.5 hours, and at least 15% A100 memory reserve. The
hard user budget is 12 hours; a failed timing gate closes the configuration
without starting the long run.

All server activity, if later authorized by this gate, is additionally limited
to `/lustre/home/users/sm2/chou/` by `platforms/REMOTE_HANDOFF.md`.
