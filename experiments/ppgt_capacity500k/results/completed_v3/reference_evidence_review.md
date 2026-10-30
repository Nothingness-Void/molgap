# PPGT capacity terminal acceptance review

## Finding

The supplied strict acceptance result remains: both arms MECHANICALLY_VERIFIED; strict_comparison is BLOCKED with error reference_artifacts and required evidence target_transform_asset. I did not rerun acceptance.

The reviewed frozen input records show a concrete schema/binding gap consistent with that blocker:

- The prepared copy of source_dataset/acceptance/experiments/ppgt_capacity500k/generated/ac5011375cfafc0d45f08887becaa65c908c6bf2c4e34150896415a199d556de/acceptance_plan.json declares the transform inline as comparison.target_transform_asset (molgap-target-transform-asset-v1, with identity, mean/std, target and source-row identities). Per-arm comparison identities also carry its identity and SHA.
- The generated ppgt_w192/plan_input.json and ppgt_w384/plan_input.json each have empty decision_state.active_reference_ids and state_at_start.reference_ids.
- The referenced anchor plans (anchors/ppgt_w192/acceptance_plan.json and anchors/ppgt_w384/acceptance_plan.json) link only a contract and the target manifest. Their contracts and training-input metadata carry scalar target_mean_eV/target_std_eV or normalization_reference values; they do not provide a reference_artifacts.target_transform_asset binding.
- No prepared JSON contains a reference_artifacts field. The package manifest includes the plan and anchor records, but no separate target-transform asset record.

So the transform exists as a generated comparison-plan declaration, while the required reference-artifact evidence/binding is absent and no reference ID is selected. This is the concrete input/schema mismatch visible in the records. The validator's implementation-level rule was not inspected, so no broader claim about its internals is made.

## Protected-role status

Both generated experiments/ppgt_capacity500k/results/completed_v3/role_audit/roles_*.json records declare official_validation, test_challenge, and test_dev as unknown. The retrieved arm role_audit.json records say those roles were untouched for both arms. These are role-audit declarations/statuses in JSON; the reviewed records do not establish operating-system or process-level file-open proof.

## Training implication

There is no indication that new training is needed to address this blocker: the missing item is reference-artifact evidence/binding, not a training result. Whether a valid existing source can supply that evidence is undetermined from the allowed records. This review does not authorize rewriting frozen plans or changing evidence.

## Scope

Read-only review in the owning checkout and the specified E: artifact paths. No code/source inspection, acceptance rerun, tests, training, inference, hashing, remote/network access, or subagents.
