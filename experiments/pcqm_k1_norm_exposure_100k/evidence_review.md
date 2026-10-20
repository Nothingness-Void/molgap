# Evidence reviewed before execution

- [Post-BN/Post-LN40-pass acceptance](../pcqm_structural_bias_norm_100k/terminal_acceptance/decision.md):
  BN0.141294461, LN0.146005922 eV; LN worse4.711461meV. Equal31240 updates;
  BN improves0.968meV over its final5 passes, LN is approximately flat.
  A single short cosine schedule does not identify absolute convergence or
  establish that a longer, slower-annealed LN recipe will recover.
- [Attribution](../pcqm_structural_bias_norm_100k/terminal_acceptance/attribution.md):
  no proven unique norm mechanism, overfit or physical-feature destruction.
  Online train versus eval development is not a fixed-state generalization gap.
- [Pre-norm pair](../pcqm_k1_prenorm_norm_100k/decision.md): Ada loses2.181843meV
  against matched pre-BN. This does not answer long-horizon Post-LN optimization.
- [Earlier interpretation correction](../pcqm_k1_prenorm_norm_100k/reference_binding/post_norm/interpretation_review_20261016.md):
  dropout is0.05; old mandatory-BN and proven-overfit explanations were withdrawn.
  Norm placement and recipe interaction remain distinct from operator age.
- RML trajectory/reference indexes were queried before design. No accepted
  matched160-pass BN/LN trajectory exists in this checkout. Existing local
  payload availability and strict-reference exclusions are not upgraded.

Post-LN addon is reused byte-for-byte from source d6058bc57fc92b6bcb336ddac13fca6846ef4d95.
Only its reviewed registration/policy-slot bindings are selectively restored;
current dynamic family projection is retained. No resistance addon is imported.

The shared SessionOutputs producer previously omitted LR/checkpoint histories
outside timing-enabled modes. This release uses the existing atomic transaction
for every pass and observes actual pre-update LR in the shared session; this is
evidence-publication maintenance, not a change to optimizer/model updates.
Process wall is measured for all modes; device time remains available only under
its explicit resource policy/actual allocation ledger. Historical traces stay intact.

Alternative explanations: seed variation, schedule interaction, statistical
normalization effects, retained representation differences and job contention.
Cheapest useful falsifier is the authorized matched long-horizon pair plus the
bounded warmup comparison, with complete curves; no new model family is needed.
