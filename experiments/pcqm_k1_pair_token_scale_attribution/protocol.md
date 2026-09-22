# PairToken 100K-to-500K attribution protocol

## Question

Did PairToken lose its mechanism at 500K, or did the previous comparison fail
to isolate scale because its exact matched60-v4 K1 prediction bundle was
missing?

## Frozen sequence

1. Train one K1 reference on the accepted fixed 500K/50K rows using the exact
   PairToken matched60-v4 optimizer, schedule, exposure, seed, precision,
   physical batch, tail policy, target transform, and selection semantics.
2. After terminal acceptance, compare the saved K1 and PairToken predictions
   row by row.  Report paired bootstrap uncertainty and error deltas stratified
   by graph size and frozen K1 error quantile.  This step trains no model.
3. Only if the aligned result still shows material-gain collapse, run the
   already-defined frozen PairToken interventions on its accepted best
   checkpoint: learned assignment, uniform assignment, diagonal-only pairs,
   and removed pair normalization.  This step performs inference only.

## Fixed contract

- data: accepted PCQM fixed 500K train plus 50K internal development;
- seed 42, deterministic FP32, TF32 disabled;
- one device, physical batch 128, no accumulation, drop-last train tail;
- AdamW, learning rate `4e-4`, weight decay `1e-5`, gradient clip `1.0`;
- epoch-indexed cosine over 60 epochs, 234,360 optimizer steps and 29,998,080
  sample presentations;
- raw-model selection on the internal development role;
- no official validation, test-dev, test-challenge, shadow, teacher, geometry,
  target residual, or prediction fusion.

## Decision rules

Round 1 has no architecture claim: it creates the absent canonical reference.
Round 2 determines whether PairToken's aligned gain and uncertainty at 500K
are actually smaller than at 100K.  Round 3 determines whether the learned
pair mechanism remains causally active after scale.  No architecture successor,
seed expansion, or full run is released by this protocol.

Every round must end with terminal hashes, native cost, role events, trace,
evidence, and a trajectory update before the next round can start.
