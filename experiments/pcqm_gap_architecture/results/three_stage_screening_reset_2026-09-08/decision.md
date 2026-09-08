# Three-stage screening reset — 2026-09-08

## Evidence

The accepted 100K/10K comparison established GraphState9 as a useful efficient
architecture: it improved fresh full-GPS controls at seeds 42, 43, and 44 by a
mean `0.006752 eV`, reduced the parameter count by about 25%, and increased
throughput by about 16%. This result remains valid.

The accepted full-scale EdgeState continuation on `molgap-desktop` commit
`e72a3db` reached official-validation Gap MAE `0.099638 eV` at epoch 30. A
later desktop report placed the in-progress GraphState full run at best
`0.118368 eV` through epoch
20. That report had not yet been integrated as accepted repository evidence
when this record was written, and the two models used different training
contracts. It therefore indicated that GraphState was not the leading
full-scale submission candidate, but it did not provide a matched causal
rejection of the GraphState mechanism.

The same-database pretraining audit also exposed a governance problem. Two
nominal scratch-through-40 jobs differed by `0.002692 eV`, while the former
material-promotion threshold was only `0.001 eV`. Repeated architecture
selection on one 10K validation role could therefore promote run noise and
progressively turn that role into tuning data.

The rejected structure and geometry pretraining jobs tested graph-level
surrogates. Hashed aggregate fragment statistics were predicted from a whole-
graph embedding, and noisy ETKDG geometry was reconstructed as whole-graph
histograms and moments. Those results rejected only those exact surrogates;
they did not test atom/edge/fragment-local hierarchical reconstruction or
clean-conditioned local geometry denoising.

## Decision

Architecture discovery was changed from a continuous Track B loop to a
three-stage promotion funnel:

1. **Track C triage — QM9-30K.** Literature mechanisms are screened as direct
   Gap predictors on one immutable 30K train-role subset with a fixed internal
   selection role. Each candidate is paired with a freshly initialized
   baseline in the same remote job. This stage uses one seed, does not read a
   test role, and can only reject a mechanism or nominate it for PCQM transfer.
   A 50K variant is not the default; it may be frozen before candidate training
   only if baseline calibration shows that 30K cannot resolve the gate.
2. **Track B transfer — PCQM-100K.** A nominated mechanism is retrained on the
   frozen official-train-derived 100K contract against a fresh paired baseline.
   The established 10K role remains the selection role. One new disjoint
   official-train-derived shadow role is frozen before the first transfer and
   read once only after the candidate is selected. Promotion requires at least
   `0.003 eV` paired improvement on the selection role, the same direction on
   the shadow role, and a credible full-scale runtime path.
3. **Desktop/full-scale confirmation.** Only one strong Track B winner is
   handed to `molgap-desktop` for an explicit compute-budget decision, full
   training, one-time official validation, and separately authorized test-dev
   submission. The server discovery agent does not launch this stage.

The already released sparse relative-value GraphState seed-42 job is retained
only to complete its evidence chain. Its terminal result does not authorize a
successor, confirmation seeds, or full training under the superseded loop.

GraphState9 remains an accepted 100K efficiency discovery. The epoch-30
EdgeState checkpoint remains the strongest accepted full-scale
official-validation evidence until a matched and accepted result replaces it.

The highest-priority literature-faithful Track C question is MolCHG-lite local
hierarchical pretraining: mask or classify real atom environments, real bond
environments, and deterministic chemistry-defined functional groups from the
corresponding intermediate states, followed by equal-compute direct-Gap fine-
tuning against scratch. This question must receive its own protocol and cannot
reuse the rejected graph-level hashed-fragment objective.

## Stop conditions

- A QM9 loss or non-material result closes only that exact implementation.
- A QM9 win without a PCQM-100K win does not transfer.
- A PCQM selection gain below `0.003 eV`, a shadow-role reversal, or a projected
  full run outside the approved budget blocks desktop handoff.
- Seeds 43/44 are never automatic; they require a separate shortlist and
  compute-budget decision.
- The official PCQM validation and test-dev roles remain outside server-side
  architecture discovery.
