# Unified Experiment Prospective Planning

`molgap-experiment-spec-v2` replaces the v1 experiment-level `prospective`
block with exactly `{"arms": [...]}`. Each declared arm has one binding with
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
