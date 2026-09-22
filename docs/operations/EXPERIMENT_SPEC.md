# Prospective experiment specification v1

`molgap.experiment_spec.ExperimentSpec(payload)` is the Stage 1 entry point.
`from_json`, `to_json`, `to_dict`, and `identity` provide strict loading,
canonical serialization, detached export, and the existing screen-policy
SHA-256 fingerprint convention. The synthetic fixture in
`tests/test_experiment_spec.py` is an exhaustive payload example.

The schema is `molgap-experiment-spec-v1`. Its terminal protocol is
`molgap-experiment-terminal-descriptor-v1`; the Stage 5 adapter described below
translates this descriptor into existing RML terminal wiring inputs.

Each ordered arm binds its scientific role, family/version, base digest,
initial state digest/seed, dataset/split/membership/row-order/usage digests,
feature schema, recipe and objective/sampler/transform identities. An empty
addon list requires `addon_semantics: baseline`; nonempty lists require
`ordered`. Array order is significant and preserved; object key order is not.
Canonical strings are immutable snapshots, independent of caller mutations.

The static, immutable registries expose `FamilyContract` and `AddonContract`.
`spec.family_contract(arm_id)` resolves the future adapter lookup without
importing model code or dispatching a callable supplied by an author.
Family version `1` is an interface version, not a promoted model generation.
GPTrans-T binds the `pcqm_gptrans_v4` recipe and K1 binds `pcqm_k1_full`.
These names point to the existing package modules. No historical experiment
identity is copied. V1 recipe overrides must be exactly `{}`: these frozen
recipes have no approved override range. Seed 42 is required.
Future overrides need a reviewed recipe version with typed bounds.

Platform `device_count` is a positive-integer resource declaration, not a
per-arm recipe restriction. Stage 1 accepts `device_count: 2`, including a
physical run declaring two independent arms under V5, and does not cap the
declaration at one device. Booleans and non-integer values are rejected.
The future runner/platform layer must validate the runtime relationship among
arm count, GPU visibility, processes, and declared device count, including
dual-arm isolation. This schema implements no runner and grants no proof of
real dual-GPU execution or runtime acceptance.

The two parameter-free `gptrans_variants.py` addons are registered:
`pair_prenorm` and `centered_logits`. Both require empty config and a source
digest; they are mutually exclusive attention replacements and GPTrans-only.
K1 additionally registers `k1_pair_value/1` with empty config; its model-only
mapping and migration boundary are in [K1_ADAPTER.md](K1_ADAPTER.md).
Other addons and arbitrary addon configuration are rejected. New combinations
need explicit registry/code review and a contract version change.

All object fields are required and unknown fields are rejected recursively;
duplicate JSON keys are rejected too. There is no author-supplied readiness
field. Prospective requirements and evidence requirements are declarations,
never proof that those requirements were met. No spec grants protected-role
access, full-run admission, execution, scientific acceptance, or replay/desktop
readiness. Those remain under existing V5/RML authorities.

This layer checks declared identity syntax and compatibility only. Digests
must be supplied for base, initialization (including random initial state),
recipes, policy and data identities. It does not read or authenticate their
underlying bytes. Family adapters must verify those bindings against actual
source, recipe values, accepted graph manifests, role history and authority,
runtime certificates, and source bundles. A structurally valid addon spec is
not executable by the frozen reference adapter, which rejects variants;
future adapter work must explicitly handle this boundary.

Family adapters, packaging, graph preflight, runners and terminal translation
must preserve the spec identity and cannot derive readiness from this
validator. This module does not modify any scientific or RML gate.

## Stage 5 terminal descriptor

`molgap.experiment_terminal.TerminalDescriptor(spec, payload)` validates and
freezes a declaration without filesystem access. `from_json(spec, text)` rejects
duplicate keys at every nesting level. `to_json()` uses the spec's canonical
JSON convention (sorted keys, compact separators, ASCII escaping, no nonfinite
numbers); `to_dict()` returns a detached copy. Input whitespace/key order may
vary; array order is preserved. Unknown fields, callables and import hooks are
not accepted.

| Object | Required fields | Optional fields |
|---|---|---|
| Descriptor | `schema_version`, `spec_identity`, `experiment_id`, `logical_run_id`, `arms` | None |
| Each arm | `arm_id`, `arm_identity`, `trajectory_id`, `run_id`, `trajectory`, `terminal`, `observed` | `trace`, `trace_source`, `canonical_trace_output`, `recovery_spec` |

The required `observed` object carries terminal facts independently of the RML
path bindings. It is descriptor metadata, not a second terminal package.
Every object below has exactly the listed fields; no readiness/replay boolean
or extensible execution hook is supported.

| Object | Exact fields |
|---|---|
| `observed` | `identity`, `terminal`, `artifacts`, `progress`, `costs`, `missing_evidence` |
| `identity` | `experiment_id`, `logical_run_id`, `arm_id`, `spec_identity`, `attempt_id`, `platform`, `family`, `recipe_identity`, `source_commit`, `source_package_sha256`, `data_identity`, `split_identity`, `feature_identity`, `target`, `initialization_identity` |
| `identity.platform` | `name`, `run_reference` |
| `terminal` | `status`, `exit_reason` |
| `artifacts` | `metrics`, `predictions`, `checkpoint`, `trace` |
| Each artifact | `status`, `locator`, `sha256`, `missing_reason` |
| `progress` | `epoch`, `step`, `samples` |
| Each entry in nonempty `costs` array | `metric`, `unit`, `value`, `status`, `reason` |
| Each observation fact | `value`, `missing_reason` |

The four identity bindings (`experiment_id`, `logical_run_id`, `arm_id`,
`spec_identity`) and platform `name` are mandatory plain values matching the
spec. All remaining identity fields, including platform `run_reference`, use
observation facts. Both terminal fields and each progress field also use facts.
A known observation is `{"value": <observed value>, "missing_reason": null}`;
an unknown observation is `{"value": null, "missing_reason": "specific reason"}`.
Reasons must be nonempty trimmed strings. Unknown facts are never populated
from the prospective spec or RML terminal package. A controller must obtain
observations from retained evidence before supplying known values.

Known identity facts must match these declaration bindings:

| Fact | Expected value |
|---|---|
| `family` | Exact arm `family` object (`name`, `version`) |
| `recipe_identity` | `canonical_fingerprint(arm["training"]["recipe"])` |
| `data_identity` | `canonical_fingerprint(arm["data"])` |
| `split_identity` | `canonical_fingerprint({"split": arm["data"]["split"], "roles": arm["data"]["roles"]})` |
| `feature_identity` | Arm `data.feature_sha256` |
| `target` | Arm `data.target` |
| `initialization_identity` | `canonical_fingerprint(arm["initialization"])` |

Known attempt IDs use the spec's safe-identifier syntax. Source commits must be
full lowercase 40- or 64-character hexadecimal IDs; source package identities
are lowercase SHA-256. Translation checks known attempts and commits against
the frozen trajectory action. The spec has no source package binding, so its
digest is syntax-checked only. Physical run references are explicit opaque
trimmed strings (for example a scheduler job reference), never derived from
logical-run IDs. They are not dereferenced or authenticated by this adapter.

Terminal status is `complete`, `failed`, `cancelled`, or `interrupted`, or a
null fact with a reason. `exit_reason` records the actual observed exit reason
as text, or remains explicitly unknown. Status does not imply acceptance or
readiness, and does not fill in an unknown exit reason. Progress values are
nonnegative integers or null facts; missing epochs/steps/samples are not zero.

Available artifacts use `status: "available"`, a repository-relative POSIX
`locator`, lowercase `sha256`, and null `missing_reason`. Translation verifies
their bytes with the existing artifact verifier. Missing artifacts use
`status: "missing"`, null `sha256`, and a nonempty `missing_reason`; `locator`
may be null or a known repository-relative path and need not exist. All known
locators obey the same traversal/symlink boundary as RML paths. Artifacts are
never downloaded or loaded as models. The trace artifact may bind raw or
canonical bytes; the optional RML trace paths retain their existing meanings.
An explicit missing artifact does not waive existing RML retained-evidence
requirements. Canonical trace outputs cannot overwrite artifact locators.

Cost `metric` is a unique safe identifier within the arm. `unit` records the
native unit without conversion (for example `DCU-hours` or `GPU-seconds`).
`measured` costs require a finite nonnegative numeric value, a nonempty unit,
and null `reason`. `estimated` costs require the same value/unit plus a nonempty
`reason` describing the estimate basis. `measurement_missing` requires null
`value` and a nonempty reason; the unit may also be null when unknown. Booleans
are not numeric measurements. Use an explicit missing entry when no cost was
retained; do not substitute zero or guess a value. This validator cannot prove
the truth of a reported measurement or estimate.

`missing_evidence` is a required array of nonempty textual reasons for other
evidence gaps; it may be empty. Field-specific absence reasons remain required
even when a broader gap is listed here. These observations remain in the
immutable descriptor and are not injected into or used to rewrite RML schemas.

`schema_version` must equal `TERMINAL_PROTOCOL`. `spec_identity` is exactly
`spec.identity`; experiment and logical-run IDs must match the spec. Every spec
arm must occur exactly once. `arm_identity` is
`canonical_fingerprint(spec_arm)`, matching the runner's arm fingerprint.
Trajectory IDs, trajectory files and terminal files must be unique. Per-arm RML
run IDs are explicit and may differ from the experiment's logical-run ID.
The spec's prospective trajectory is experiment-level context: the adapter does
not invent per-arm trajectories or derive their IDs from that context.

All six path fields use explicit repository-relative POSIX file paths, resolved
against the supplied `repo_root`. Absolute paths, URIs, backslashes, traversal,
empty/dot components, Windows device/stream names and symlink escapes are
rejected. Input files must exist when translating; the output need not exist.
Optional paths are omitted when unknown (not null or placeholder text).

`translate_terminal_descriptor(repo_root, spec, descriptor)` returns the exact
arm sequence accepted by `close_terminal_multi_arm`, in descriptor order.
`arm_id` becomes `arm_identifier`; identity and observed metadata are removed. Explicit
path strings are retained, except `recovery_spec`: its path stays in the
descriptor while its strict JSON object is loaded into the closure input,
as required by the existing API. Nothing missing is filled in.

The recovery object requires `metric_semantics` and permits only
`trajectory_id`, `run_id`, `rows_key`, `field_mapping`, `device_time_semantics`
in addition. Supplied IDs must match the arm. Mapping keys use existing trace
`FIELDS` and values are explicit source column names. Metric semantics use the
existing trace validator; each non-null definition has exactly `metric`,
`unit`, `target`, `role_identity`, `weights`, `direction`. Unknown metric
semantics remain null. No parser/import/callable configuration is supported.

Translation uses existing trajectory/trace schema validators, verifies available
artifact hashes, checks terminal format and trajectory/run/action cross-links,
and rejects shared finalization
directories and output/input collisions. It does not recover traces, discover
missing evidence, verify scientific acceptance, finalize records, rebuild RML,
run models, alter the production registry or grant READY/replay authority.
Fingerprints bind declarations; they do not prove artifact provenance.

`execute_terminal_descriptor(repo_root, spec, descriptor)` is the separate thin
call to `close_terminal_multi_arm(repo_root, translated_arms)`. Only this
explicit operation enters existing trace recovery, finalization and pipeline
side effects. Retained-trace guards, artifact verification, acceptance and
readiness remain with those existing authorities. Successful translation is
not proof closure will succeed; omitting `trace` cannot waive retained-trace
evidence. Multi-arm execution retains the existing per-arm transaction boundary,
not an all-arms atomic transaction.

Example for an already validated spec and retained per-arm RML files (the
mapping is supplied by the controller, never inferred by the adapter):

```python
from molgap.experiment_spec import TERMINAL_PROTOCOL
from molgap.experiment_terminal import TerminalDescriptor, translate_terminal_descriptor
from molgap.screen_policy import canonical_fingerprint

# bindings maps every spec arm_id to explicit trajectory_id, run_id,
# trajectory, terminal, and the complete observed object described above,
# plus any of the four optional path fields. Observations come from evidence;
# copying prospective identities alone does not establish an observed fact.
declaration = spec.to_dict()
payload = {
    "schema_version": TERMINAL_PROTOCOL,
    "spec_identity": spec.identity,
    "experiment_id": declaration["experiment_id"],
    "logical_run_id": declaration["logical_run_id"],
    "arms": [
        {"arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
         **bindings[arm["arm_id"]]}
        for arm in declaration["arms"]
    ],
}
descriptor = TerminalDescriptor(spec, payload)
arms = translate_terminal_descriptor(repo_root, spec, descriptor)  # read-only
# Example translated arm without optional evidence:
# {"arm_identifier": "gptrans_t",
#  "trajectory": "experiments/question/gptrans_t/trajectory.json",
#  "terminal": "experiments/question/gptrans_t/terminal.json"}
```

Focused synthetic coverage is in `tests/test_experiment_terminal.py`.
