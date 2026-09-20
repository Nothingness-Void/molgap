# PCQM scale-transfer attribution protocol

## Question

Why have the bounded PCQM architecture mechanisms that passed a 100K screen
usually failed to retain a material gain at 500K?

## Evidence boundary

This is a no-training, no-inference evidence synthesis. It reads only existing
accepted contracts, decisions, trajectories, and machine summaries from the
desktop branch and the fetched `molgap-server` commit recorded in
`trajectory.json`.

The primary causal set is limited to mechanisms with a frozen 100K reference
and a later matched 500K attempt. Historical runs with changed feature schemas,
evaluation roles, optimizers, EMA semantics, or incomplete convergence are
reported separately as contextual evidence.

No official validation, test-dev, or challenge role is read. No remote job is
submitted by this analysis.

## Outputs

- `analysis.json`: machine-readable transfer table and confounders.
- `decision.md`: attribution, limits, and the minimum protocol repair.

