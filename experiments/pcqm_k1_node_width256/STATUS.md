# Status

Candidate implementation and CPU initialization qualification completed.
[Result](qualification_result.json): 6,035,201 parameters, 1.649495 times the
3,658,817-parameter reference; finite state, exact save/load state identity,
and preserved width192 default initialization. Source commit:
`d6d84c138f86b536a9e60d96a095b345a9b61669`.

[Preparation](preparation_report.json) passed source syntax, selected-checkout
imports, candidate-arm schema and the family-owned frozen recipe check.
The [candidate arm](candidate_arm.json) and [recipe](training_recipe.json) are
retained preparation inputs; no complete training Spec/prospective or release
package has been published. GPU/real-shard qualification remains pending.

No width-256 training submission or molecular-data access has occurred.

Training release is pending a qualified strict retained K1-192 reference.
The old reference lacks checkpoint identity; the separately submitted K1/SSMA
reference must be accepted before binding it here. Its retained observation
reported RUNNING; the fresh SDK status query returned HTTP403, so current
remote state is UNKNOWN. See [protocol](protocol.md).

Keep this experiment ACTIVE on its owning branch. Next action: reconcile the
existing reference attempt, inspect its accepted trace/runtime/role/cost inputs,
then freeze and prepare the one-arm width256 training Spec through the modular
workflow. No baseline retraining or release-gate bypass is authorized.
