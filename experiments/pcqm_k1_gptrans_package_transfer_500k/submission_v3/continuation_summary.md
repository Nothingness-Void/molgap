# Version3 continuation

Kaggle1 accepted version3 of the same kernel137144136; UI script version
355422526. Exact response, source pull, input manifest pins and immutable
launch receipt are retained here. Same Spec and prospective records; executable
source0d610f0d only adds log visibility, bounded bootstrap waits and explicit
resume transport/runtime pins. No model/recipe/role/precision change.

[Training start observation](training_start_observation.json) confirms CPU
validation, both isolated training-only resume preflights and actual resumed
training: K1 epoch7 at step23,437; GPTrans epoch10 at step35,155, followed by
its100th batch at step35,254. Worker output is both retained and forwarded;
training emits progress every100batches or30seconds at a batch boundary, plus
phase status every30seconds. This does not imply terminal acceptance.

The focused regression batch passed13tests once. Shared release checks passed
and were repeated by the existing accelerator submitter before POST. Both
remote resume manifests/source pins match. RML check --frozen --portable
passed after committing the actual submission records. No successor question,
automatic monitor, server handoff or shutdown was requested for this continuation.

The prior user-cancelled v2 scheduler cost is4.941222allocated native T4hours.
This invocation is bounded to9hours/T4x2. Reconcile terminal outputs and
measured allocation before a further continuation, then accept each arm
independently under the frozen protocol.
