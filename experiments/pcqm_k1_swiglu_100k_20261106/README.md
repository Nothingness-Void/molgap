# K1 node SwiGLU — design and implementation preparation

Status: design/code prepared; runtime qualification and execution are not authorized.
No model construction, forward/backward, checkpoint inference, training, source
publication or remote submission was performed. This worktree is independent of
the GPTrans query-readout experiment.

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
a new running worker. No physical notebook layout is released here. If a later
execution request interprets even arms as newly launched workers, resolve that
layout before submission; do not silently launch an odd-worker job or retrain A
to fill a slot. No unrelated second candidate is invented.

## Implementation and reuse

`src/molgap/k1_swiglu_ffn.py` plus its single catalog entry implement the delta.
The existing K1/3 family loader strictly loads the original initialization first;
the composed addon then uses `ModelSubstrate.replace_submodule` on only
`local_blocks.0.mlp` through `.8.mlp`. The common session, optimizer, EMA, source
packaging, platform submission, strict acceptance and RML owners are unchanged.

`candidate_recipe.json` was built with the existing `build_family_recipe` metadata
API. It is a draft recipe, not an execution authority, frozen Spec or submission.
No editable account/key/budget authority was invented. Its reference is pinned
by the paths in qualification.md, rather than copying retained scientific evidence.

Do not run `prepare-family-training` or the submit front door as a harmless
metadata check: preparation executes a model sandbox. Before any such diagnostic,
publish a canonical prospective trajectory through the existing RML owner and
obtain applicable diagnostic authority. No prospective trajectory is published
by this planning-only change; no ACTIVE training record is invented.
