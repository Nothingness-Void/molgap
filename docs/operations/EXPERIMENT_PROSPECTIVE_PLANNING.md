# Unified Experiment Prospective Planning

`molgap-experiment-spec-v2` replaces the v1 experiment-level `prospective`
block with `{"arms": [...]}` and an optional `same_run_replay` declaration.
Each declared arm has one binding with
exactly `arm_id`, `trajectory_id`, `plan_spec_ref`, `plan_spec_sha256`, and
`output`. The arm IDs must match the spec arms one-to-one. V1 remains readable
for its existing diagnostic, package, preflight, and terminal paths, but cannot
use `plan-prospective`.

Prepare one existing RML `plan()` input JSON per arm, using the normal RML
planning schema and supplying its scientific hypothesis, evidence pointers,
role/budget state, decisions, actions and prospective cost events explicitly.
The input's `trajectory.trajectory_id` must equal the binding's trajectory ID.
Its `trajectory.state_at_start.source_config_identity` must equal
`canonical_fingerprint(full_spec_arm)`, where `full_spec_arm` is that arm's
entire declaration, not the experiment-level spec. Do not invent missing
scientific content or collapse two arms into one trajectory. Bind
`plan_spec_sha256` to the exact JSON file bytes, including whitespace.

`plan_spec_ref` is a repository-relative POSIX `.json` file path; `output` is
a new repository-relative directory below `experiments/`. Absolute paths,
drives, traversal, symlinks/junctions/reparse points and existing outputs are
rejected. Use a canonical no-newline ExperimentSpec JSON file:

```powershell
& .venv\Scripts\python.exe -m molgap.experiment_cli plan-prospective `
  --spec C:\path\to\spec-v2.json --repo-root C:\path\to\molgap
```

The command reads and checks every plan input before publishing anything. It
calls RML `plan_many()` once, giving each arm the same pre-publication evidence
snapshot, verifies the returned order and identities, then rebuilds RML derived
indexes. Success means only that prospective planning and rebuild finished. It
does not submit a job, authorize training, create evidence, use an evaluation
role, or create READY_FOR_DESKTOP.

When the Spec explicitly declares
`"same_run_replay":{"reference_arm_id":"...","candidate_arm_ids":["..."]}`,
the planner also freezes a `same_run_replay` binding in those prospective
trajectories. It binds the Spec identity, logical run, arm roles, and reference
trajectory path before submission. The input plan files must not supply or
override this binding. The listed arms must have matching `scientific_role`
declarations and must be published in the same planning batch. Other arms remain
independent.

If `plan_many()` stops after publishing some arms, the nonzero result lists
`completed_records` and requires reconciliation; it neither deletes them nor
rebuilds RML. A rebuild failure also returns nonzero while retaining all
published canonical plans. Do not rerun with the same output paths or treat
either result as submission-ready. Resolve the canonical records and rebuild
state explicitly before any subsequent submission decision.

For v2 terminal translation, each descriptor arm's trajectory ID must equal
its prospective mapping, and the retained prospective trajectory must still
carry the mapped full-arm fingerprint in `state_at_start.source_config_identity`.
V1 terminal translation is unchanged.

For a same-run replay pair, terminal translation also checks the frozen Spec
binding and one observed platform run, attempt, source commit, and source
package identity. Each replay-eligible arm's terminal package must carry
`same_run_observation` with schema `molgap-same-run-observation-v1` and the
Spec identity, logical run ID, platform name/run reference, attempt ID, source
commit, and source package SHA-256. These values must match the descriptor's
observations; the reference and candidates must match each other. Direct RML
closure also checks the retained observation before admitting an eligible
trace. Execution closes the reference before its candidates. A
reference trace may bind its own accepted V5 evidence ID; an eligible candidate
trace must bind that exact accepted reference ID and carry strict V5 comparison
readiness. Both require explicit, calibrated trace manifests. The closure
default remains ineligible. An ineligible or incomplete arm may still have a
valid terminal result without a replay-pool entry. Existing frozen trajectories
without the new binding retain their original qualification state.
