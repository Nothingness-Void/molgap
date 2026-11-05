# K1 node SwiGLU — design and implementation preparation

Current release/job state: [STATUS](STATUS.md). The user's subsequent execution
approval released necessary qualification and one fixed100K candidate run.
The [execution qualification](execution_qualification.md) records the real sandbox
failure, infrastructure repair, corrected pass and submission. This worktree is
independent of the GPTrans query-readout experiment.

## Decision

Retain one narrowly defined100K hypothesis: can a matrix-budget-matched gated
node FFN improve the unmodified K1 backbone? This is not yet a positive result,
a new training release or an accepted performance optimization. There is no
observed FFN deficiency proving that K1's500K plateau comes from this module.
The experiment would test that alternative, not assume it.

The [A/B/C review](research_review.md) records the proposal, paper-based objection
and bounded decision. [Protocol](protocol.md) owns the intended comparison and
recipe. [Qualification](qualification.md) owns checks completed and remaining.

Exactly two scientific comparison arms are planned:

| Arm | Model | Intended treatment |
|---|---|---|
| A reference | Original K1, no scientific addon | Reuse the retained strictly qualified baseline; no default reference retraining |
| B candidate | Same K1, nine node FFNs replaced with SwiGLU | Future authorized100K candidate training only |

A retained comparator is part of the two-arm scientific comparison, but is not
a new running worker. The reviewed design was approved for execution: only B is
newly trained, while A is reused. The platform retains its T4x2 allocation shape;
this is not two newly trained workers. No unrelated second candidate is invented.

## Implementation and reuse

`src/molgap/k1_swiglu_ffn.py` plus its single catalog entry implement the delta.
The existing K1/3 family loader strictly loads the original initialization first;
the composed addon then uses `ModelSubstrate.replace_submodule` on only
`local_blocks.0.mlp` through `.8.mlp`. The common session, optimizer, EMA, source
packaging, platform submission, strict acceptance and RML owners are unchanged.

`candidate_recipe.json` was built with the existing `build_family_recipe` metadata
API. The authorized input is experiment.json; it binds the retained reference's
exact canonical recipe bytes and the baseline projection of its original frozen
Spec in data_declaration.json. The candidate draft's initial metadata alone did
not authorize execution. Reference scientific evidence stays with its original owner.

Preparation is not a metadata-only check. Execution used the published prospective
record and applicable authority, existing sandbox and prepared-run submitter.
The [initial qualification snapshot](qualification.md) remains historical;
it is superseded by the execution report for actual CPU checks. Native runtime
qualification and final replay-ready scientific closure remain separate.
