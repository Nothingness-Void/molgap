# RML lifecycle implementation contract

Reusable APIs live in `src/molgap/research_memory/`. These commands consume
metadata and retained artifacts, never dispatch work or load a model. The
existing V5 validators and strict comparability grouping remain authoritative.
No policy or historical backfill is installed by this implementation.

Filesystem arguments and locally resolved pointers are confined to the selected
repository root by `paths.py`; explicit `..` traversal and symlink escapes are
rejected before access. Remote metadata URIs remain pointers, not filesystem
paths. The recovery API accepts `repo_root=` (default: the configured repository).

## Canonical trace and recovery

`trace.py` exports `canonicalize_trace`, `validate_canonical_trace`,
`load_canonical_trace`, `trace_digest`, and `RMLTraceRecorder`.
`schemas/trace-v1.schema.json` describes the new artifact. Existing manifests
remain v1; optional `trace_artifact_sha256` binds exact artifact bytes.
`trace_digest(object)` hashes the canonical JSON serialization; passing a path
validates the trace and hashes its exact bytes. Use the latter for existing files.

Every observation has a contiguous zero-based `sequence`, an `event`
(`observation`, `checkpoint`, `resume`, `terminal`), and every declared field
present with either an observed value or null. Metrics have explicit metric,
unit, target, role_identity, weights (`live`/`ema`), and direction semantics.
Device timing, when present, declares `device_time_semantics=sum_over_devices`;
cumulative counters cover the whole run, including resumes. Nothing is derived
from epochs, batch size, learning-rate schedules or terminal scores.
Unknown top-level/observation fields, repeated non-null optimizer/sample axes,
and reused checkpoint identities are rejected by the shared validator and
recorder. There may be at most one terminal event, and it must be last; an
unfinished or partial trace may have none.

Constructing a recorder begins or reopens a run with exact identity/semantics.
`append_observation`, `checkpoint_event`, `resume_event`, and `terminal_event`
each validate and atomically flush a complete snapshot. Use one writer per path.
An error propagates; the in-memory snapshot is not advanced on a failed write.
POSIX flushes directories as well as files; standard-library Windows provides
atomic replacement and file fsync, but not a directory-fsync power-loss guarantee.

`recover-trace` requires one or more explicitly ordered CSV, JSON-array or JSONL
sources. Its spec contains `trajectory_id`, `run_id`, `metric_semantics`, optional
`device_time_semantics`, and `field_mapping` (canonical field -> source column).
A JSON object source also needs `rows_key`. Recovery preserves source order,
marks missing fields null, records per-source SHA256/row ranges/field coverage,
and refuses to overwrite output. It never creates checkpoint identities or
terminal events from a terminal metric. No batch historical recovery is implied.

## Prospective planning

The plan spec contains `trajectory`, `decision_state`, and `costs`. Scientific
content and actions come from the caller. The trajectory follows the existing
prospective schema; `owner` must explicitly be `desktop` or `server`. Schema,
record mode, trajectory/hypothesis IDs and empty result may be omitted.
Output is a new experiment directory; existing
directories are never overwritten. Referenced contracts, decision authority,
budget, and role metadata must already exist locally.

Supply `decision_state.available_actions`, `chosen_action`, `policy_id`,
`policy_version`, and an explicit `state_timestamp`. The planner binds HEAD,
known trajectory/evidence IDs, chosen references, policy bytes, and source
hashes. It persists `decision_state.json` and `policy_snapshot.json` beside the
trajectory. Later compilation does not expand this knowledge snapshot.
Referenced prospective cost events must be supplied, with estimated or missing
measurements rather than invented measured costs. No role-access event is
created from a role plan.

For scale/research action replay, supply `action_inputs_ref` while planning.
That retained metadata contains trajectory_id, run_id, state_timestamp,
policy_type, evidence_ids, reference_id, the existing comparability_identity,
observables, and optional hardware array. The planner freezes its hash and
rejects evidence IDs outside the knowledge snapshot.

## Terminal ingestion and publication

`finalize` takes an original prospective trajectory path or experiment directory,
a retained `terminal.json` or its directory, and optional canonical trace path.
The terminal-package schema describes the structured input. Its required fields
are format (`molgap-rml-terminal-package-v1`), trajectory_id, run_id, action_id,
finalized_at, acceptance_ref, artifact_hashes, evidence, decision, costs and roles.
Use an empty array for unavailable observed cost/role records. The supplied
timestamp is part of the immutable input identity, not generated on each retry.

The retained acceptance metadata must explicitly contain the same evidence_id,
run_id, V5 outcome object, and trajectory_decision. An existing adapter may
export this metadata after its no-inference acceptance; finalize does not guess
the shape of a runner's logs. `artifact_hashes` covers all evidence artifacts,
authority pointers, decision authority, acceptance and observed cost/role sources.
All new evidence artifacts must be locally retrievable and hash-verifiable.
Remote URIs alone cannot establish a new finalization; retain their evidence
metadata locally first. Finalize never re-evaluates predictions.

Each cost/role event must occur exactly in the `costs`/`roles` array of its
hash-bound evidence_ref metadata. Missing cost input produces one explicit
measurement_missing event with unknown hardware/platform/attempt. Missing role
input creates no observed role event. An eligible canonical trace also requires
hash-bound `comparison_readiness_ref` passing the existing strict V5 validator;
the supplied v1 trace_manifest keeps its contract and reference identities.
Evidence, terminal-package and acceptance `role_use` claims are checked against
source-verified role events: consumed/read/used requires matching observed access,
and untouched/not_applicable conflicts with any observed access for that role.

The transaction stages under ignored `research_memory/.staging/`, validates
record shapes, cross-links, trace and publication hashes, then renames one
directory to `experiments/<question>/rml_finalized/`. That directory contains
terminal trajectory, evidence, events, optional trace/manifest, the original
prospective bytes, terminal input, and a finalization receipt. Discovery verifies
the receipt and overlays the explicitly replaced records; originals remain
untouched. Consumers should use RML discovery rather than glob duplicate IDs.
Same original trajectory and input digest returns ALREADY_FINALIZED. Changed
inputs fail closed. Interrupted staging is invisible to discovery; a failure
after the rename may leave a complete committed package, retrievable on retry.

An optional `action_replay` binds inputs_ref/inputs_sha256 and truth_ref/truth_sha256.
Inputs must match the prospectively frozen digest. Truth names terminal result
evidence IDs and matching policy_type. Scale truth explicitly supplies
`label={winner: bool, promotion_passed: bool}`; it is not inferred from a small
screen winner. Research truth supplies `correct_actions`. Optional `action_costs`
maps actions to explicitly measured saved_device_hours, added_device_hours and
hardware; absent counterfactual costs remain unknown.

## Replay definitions

The pool admits canonical candidate/reference pairs only after the existing
strict grouping gate and local artifact/hash validation. Removing an unavailable
reference removes that replay world. Partial traces remain labeled partial;
unavailable or incompatible traces receive explicit exclusions. Scale/research
entries instead use the separately frozen action inputs and truth; these cannot
serve as canonical training-prefix entries.
Candidate trace references must belong to the trajectory's frozen reference IDs.
Reference traces bind their own accepted evidence identity in terminal results.
Retrospective partial trajectories remain `historical_partial` even when their
retained trace reaches terminal exposure; coverage does not grant prospective
authority.

Screen/early-stop policies observe the first record at an exact optimizer-step
or sample-presentation cutoff. Missing cutoffs or observables produce action=null,
not STOP. The rule function receives only its required fields from that record.
Terminal truth is accessed after the decision. No interpolation, best-of-future
selection, or later record at the same cutoff is used. `PROMOTE` retains a screen;
only a separate scale_up policy emits SCALE. Rule thresholds are caller-supplied.

Screen labels require agreement between frozen trajectory and evidence:
POSITIVE_UNDER_CONTRACT means winner/promotion-pass; NEGATIVE_UNDER_CONTRACT means
non-winner/promotion-fail. Other labels are excluded, including infrastructure,
NO_TRAIN, STOP_FOR_COST and INCONCLUSIVE. This conservatively leaves ambiguous
historical positive-below-gate labels unscored rather than reinterpreting them.

- evaluated_decisions counts labeled actions other than null and DEFER.
- false_stop_count counts STOP on a terminal winner; its rate divides by labeled
  STOP decisions. slow_starter_count is the same count for prefix policies only.
- false_promote_count counts PROMOTE/SCALE failing its frozen promotion truth;
  its rate divides by labeled PROMOTE/SCALE decisions.
- winner_retained_count counts winners receiving CONTINUE/PROMOTE/SCALE. Its rate
  divides by all labeled winners in the report, including undecided winners.
- Savings for STOP use measured cumulative device-time differences to the
  observed terminal endpoint. Continuing the completed run saves/adds zero only
  when native cost is known. Scale/research costs need explicit measured action
  costs. Measured, estimated, measurement_missing and not_applicable records
  remain distinct in the pool. Only an entirely measured native cost has a
  numeric total; missing/not_applicable never becomes zero or contributes to
  saved/added totals. unknown_cost_count counts evaluated
  decisions with incomplete cost. Net hours = saved minus added.
- Costs aggregate only within one hardware unit; any missing measurement makes
  that total null. Mixed hardware has separate buckets and no scalar total.
- Empty denominators yield null with reasons. Research-action reports use action
  match counts/rates; scientific confusion metrics are inapplicable/null.

`terminal-pipeline` / `finalize_rebuild_backtest` performs finalize, validation,
rebuild (including policy replay), and returns structured pool/report deltas with
next_decision=SOL_REQUIRED. A derived-update failure preserves the complete
finalization and is retryable. It never messages a controller, approves a policy,
or launches a successor. Derived files are individually atomically replaced and
deterministically reproducible; they are not the canonical transaction boundary.

For an already completed remote run, use the existing acceptance adapter,
`recover-trace`, `close_terminal_multi_arm`/`terminal-pipeline`, `validate`,
`rebuild`, and `check --frozen`. Retain only the hash-bound decision artifacts
required by those commands (including the selected checkpoint, aligned
predictions, trace, manifests, source identity and acceptance metadata). Do not
download every epoch checkpoint/prediction or write a one-off terminal adapter
when these entry points already cover the run. Reconcile the frozen contract
against observed source, optimizer, schedule and role identity before assigning
strict comparison status. A contract mismatch remains visible as a terminal
anomaly; a valid trace alone does not grant strict replay eligibility. Never
rewrite a frozen contract or discard physical results to make the record pass.

## Desktop integration and authority

The lifecycle infrastructure is selectively ported from `molgap-server` commit
`209a7349` onto Desktop base `15719329f0af73700e8f2967c8e24bfa000eeed7`.
Canonical schemas and replay semantics are shared. Planning requires explicit
ownership; finalization preserves that ownership and the prospective snapshot.
Neither operation grants permission to execute a declared action.

Desktop retains its existing prelaunch/reference validation and
`src/molgap/v5_desktop.py` as the authority for full-scale admission and protected
role access. Terminal ingestion consumes already accepted metadata; it does not
perform official validation/test evaluation. READY_FOR_DESKTOP remains a
server-owned evidence package requiring a separate Desktop decision. Neither
READY, replay actions such as SCALE, nor pipeline COMPLETE authorizes a full run.
The pipeline always leaves `next_decision=SOL_REQUIRED`; it has no submission,
heartbeat, takeover, or cross-machine controller integration.

Each checkout discovers and rebuilds only its own selected repository corpus.
This port includes no Server trajectories, evidence, policy instances, or derived
data. The existing Desktop derived files are retained unchanged. At the user's
request, no tests, RML validation, acceptance, or rebuild were run for this port;
Luna owns testing and regeneration of Desktop replay/policy derived outputs.
