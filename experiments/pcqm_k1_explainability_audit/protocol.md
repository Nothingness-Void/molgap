# Frozen K1 V4 explainability audit

## Question

Before another architecture screen, where does frozen K1 fail and where do
the already-closed K1 variants help or hurt on exactly the same molecules?

## Stage 1: paired error map

- Join all available same-contract K1-v4 development prediction payloads to
  the accepted fixed PCQM 100K graph cache by `source_idx`.
- Verify all 50,000 targets and row identities exactly.
- Slice paired absolute-error changes by graph size, bond density, OGB atom and
  bond categories, RWSE summaries, and target Gap.
- Report paired normal-approximation intervals within each slice and the non-deployable oracle
  headroom across existing variants.
- Do not construct a model, train, read official validation, or read test roles.

Stage 1 is diagnostic only. It cannot promote a variant. Its output decides
whether a separate frozen-checkpoint causal audit is informative.

## Stage 2: conditional causal audit

Only after Stage 1 identifies a coherent failure stratum, use the frozen K1
checkpoint on a deterministic subset to measure:

1. layer-3, layer-6, and layer-9 exchange ablations;
2. assignment entropy/effective atom count and update-to-hidden norm;
3. node-state dispersion before and after each exchange;
4. atom-order permutation invariance.

Attention/slot assignments are not treated as causal explanations by
themselves. No new model is trained in this stage.

## Decision gate

A successor architecture may be proposed only if both conditions hold:

- one interpretable molecular stratum shows a consistent paired deficit; and
- a frozen-model intervention identifies one information-flow bottleneck tied
  to that stratum.

Otherwise K1 architecture discovery pauses instead of opening another
micro-variant. Any later candidate still needs a distinct V4 seed-42 protocol,
candidate preflight, and the unchanged `0.003 eV` promotion gate.
