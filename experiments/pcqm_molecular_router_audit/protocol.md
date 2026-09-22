# Frozen protocol: Molecular Router feasibility audit

## Question and scope

Can an already-completed PCQM model that is globally weak act as a specialist
on an identifiable molecular subspace?

This run is limited to Phase A-C:

1. audit and align existing prediction payloads;
2. measure pairwise and greedy multi-expert Oracle headroom;
3. test whether expert wins have recognizable chemical, scaffold, fingerprint
   cluster, or prediction-disagreement structure.

It performs no training, model inference, Router fitting, protected-role read,
remote submission, or architecture change. Oracle results are label-aware and
non-deployable.

## Roles and base

- Dataset: accepted fixed PCQM4Mv2 100K screen.
- Analysis role: the 50,000-row official-train-derived development range
  `[100000,150000)` only.
- Base: immutable `neural_atom_k1_v4` prediction payload.
- Source rows: the locally retained official training CSV, used only to recover
  the aligned SMILES and training target for those source indices.
- Official validation, test-dev, and test-challenge remain unread.

Prediction payloads must match the base `source_idx` and `target_eV` tensors
exactly. Duplicate predictions are collapsed by prediction SHA-256. K1-lineage
V4 endpoints form the primary pool. GPTrans endpoints and the known
PairToken+MoSE feature-identity mismatch may be reported only in a separate
contextual endpoint pool; they cannot strengthen a strict K1-contract claim.

The role has already been used for architecture selection. Therefore all
findings are hypothesis-generation evidence, not an unbiased estimate of a
future Router's performance.

## Frozen Oracle gate

The primary K1-lineage pool passes the capacity portion only if at least one
pair satisfies all of:

- Oracle gain versus K1 is at least `0.005 eV`;
- expert win rate is at least `5%`;
- mean winning margin is at least `0.010 eV`;
- Oracle gain is positive in all five deterministic source-index folds.

The greedy multi-expert Oracle must additionally improve K1 by at least
`0.010 eV`. These thresholds are deliberately larger than a normal screen
margin because a practical Router can recover only part of a label Oracle.

## Frozen structure gate

Only the five primary-pool experts with the largest pairwise Oracle gains are
examined for routing regularity. A specialist is structurally identifiable
only if both conditions hold:

1. at least one inference-visible scalar signal (absolute base/expert
   disagreement or an RDKit descriptor) has average-precision lift of at least
   `0.05` over the expert-win prevalence; and
2. at least one unsupervised Morgan-fingerprint cluster has at least 500 rows,
   expert-win uplift of at least `0.10`, and positive uplift in every one of
   the five deterministic folds.

Scaffold enrichments are descriptive only because repeated scaffolds can be
sparse. Embeddings, seed variance, and train-set OOD distances are reported as
unavailable unless an aligned frozen asset is actually present.

## Decision semantics

- `GO_TO_ROUTER_BASELINE`: asset, Oracle-capacity, multi-expert, and structure
  gates all pass. This authorizes only a separately frozen Phase-D protocol;
  it does not authorize training in this experiment.
- `NO_GO_STRUCTURE_NOT_LEARNABLE`: Oracle capacity passes but no top specialist
  passes the structure gate.
- `NO_GO_INSUFFICIENT_ORACLE`: the capacity gate fails.
- `INCONCLUSIVE_ASSET_FAILURE`: identities or minimum assets fail.

No gate may be changed after reading the audit result.

