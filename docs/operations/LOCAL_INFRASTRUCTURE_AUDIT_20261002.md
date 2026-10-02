# Local experiment infrastructure audit

Audit date: 2026-10-02 (Asia/Tokyo). Inspected source commit:
`5a85bd446a131b5b365766af1efe460911ce9ffb` on `molgap-server`.

The requested repairs were recorded in the
[2026-10-03 completion review](LOCAL_INFRASTRUCTURE_COMPLETION_20261003.md).
The findings below describe the inspected commit before those repairs.

## Assessment

The audited infrastructure had coherent ownership and reusable components for
experiment identity, prospective planning, immutable packaging, local release
checks, isolated family execution, retained-output inspection and RML closure.
Supported variants could reuse this chain without another packager or finalizer.
This supported a positive modularity assessment, but did not establish one
complete, uniformly supported or interruption-safe submission path.

Two release/recovery gaps warranted priority fixes before relying on the
registered Kaggle workflow for unattended training. Its generated configuration
was internally assembled from validated inputs; the findings below did not
demonstrate that every generated preparation was defective. They demonstrated
missing guarantees at its final release and cross-instance recovery boundaries.

No training, inference, protected-role access, platform API operation,
prospective publication in this repository, or scientific-contract change
occurred during this audit. Production source was unchanged.

## Ownership and route selection

| Concern | Reused owner | Assessed scope |
|---|---|---|
| Immutable identity and per-arm plan | `experiment_spec.py`, `experiment_prospective.py`, `research_memory/plan.py` | Shared; deep all-arm planning validation and explicit frozen authority |
| Source and upload inputs | `experiment_package.py`, `experiment_staging.py` | Shared; committed source allowlist and separately pinned input bytes |
| Server variant preparation | `prepare-release`, `check-preparation` | Existing server contracts, explicit template and upload configuration |
| Registered family preparation | `prepare-workflow`, `experiment_execution.py`, `kaggle_workflow.py` | K1 v2 and registered GPTrans modes; Kaggle T4 allocation only |
| Family execution | `k1_screen_training.py`, `gptrans_screen_workflow.py` | Owning recipe, loader, optimizer, selection and checkpoint logic |
| Platform submission | Owning workload skill and platform adapter | Separate account, scientific/budget release, POST and reconciliation |
| Retained outputs and closure | `experiment_family_workflow.py`, `experiment_terminal.py`, RML terminal wiring | Mechanical inspection followed by independent scientific/role/cost/replay gates |

The two preparation routes deliberately preserved different capabilities.
The four checked-in server GPU Specs contained eight arms, all outside the
registered execution addon set. They remained routed through the server
`prepare-release` and owning adapters. Their rejection by
`experiment_execution.training_adapter` was expected, not a loss of server
support. Neither registered execution nor its preparation registry supplied
IMS/SCNet adapters or a generic EdgeState epoch loop.

For K1 v2/SSMA and the registered GPTrans modes, use
[registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md). For server
author/input/pair-scale and EdgeState primitive wiring, use
[server workflow](EXPERIMENT_WORKFLOW.md). Declaration support alone does not
choose an executable route.

## Findings

### P1 — Final launch checks omitted executable package and job bindings

Owner: `experiment_preflight.py:869` (`check_workflow_binding`), called by
`check_release_inputs` and the owning Kaggle adapter's before-POST recheck.

The helper checked Spec identity, platform metadata, prospective file digests
and the entrypoint's launch digest. It did not compare the launch's
`expected_package_identity` or `expected_source_archive_sha256` with the
verified source package. It also did not validate launch `jobs` with the shared
`validate_execution_plan` or require the launch format and complete schema.

An isolated synthetic source repository, tiny trusted CPU initialization
states, packaged recipes and staged sidecars were passed through the actual
package builder, release checker and `_verify_release_report`. Each changed
launch was freshly hashed and its entrypoint repinned before checking:

| Frozen launch case | Local release result | Before-POST recheck |
|---|---|---|
| Control | `LOCAL_RELEASE_INPUTS_VERIFIED` | Passed |
| Wrong expected package identity | `LOCAL_RELEASE_INPUTS_VERIFIED` | Passed |
| Wrong expected source archive SHA256 | `LOCAL_RELEASE_INPUTS_VERIFIED` | Passed |
| Both jobs assigned to one device | `LOCAL_RELEASE_INPUTS_VERIFIED` | Passed |
| Missing `jobs` | `LOCAL_RELEASE_INPUTS_VERIFIED` | Passed |

The remote bootstrap/runtime would reject these configurations after a worker
was provisioned. The normal preparation builder populated these fields
correctly, and an unrepinned change after preparation was caught by the existing
byte checks. The gap affected validation of newly frozen malformed launches,
not the immutability protection of an already accepted report.

Suggested repair: reuse `validate_execution_plan` in final local binding,
require the launch schema, compare package/archive identity with the already
verified manifest, and bind each job's recipe/initialization paths to the same
release inputs. Exercise these adversarial cases through the actual before-POST
checker with no network call.

### P1 — Registered workflow lacked a cross-instance resume transport

Owners: `experiment_workflow.py:222`, `kaggle_workflow.py:45`,
`platforms/kaggle/run_experiment.py:59`, and the family trainers.

Preparation accepted only an initial state per arm. Staging copied source
sidecars and those initial states. The bootstrap created a fresh source tree
and sent training output to `/kaggle/working/experiment`. Neither the plan,
launch nor bootstrap provided a hash-bound retained-resume input and restoration
step. K1 and GPTrans resumed only if `last_checkpoint.pt` and the corresponding
retained state were already present in their output directory.

This established same-directory resume capability, not recovery on a fresh
Kaggle instance. The owning platform's existing manual recovery routes remained
separate; they were not composed by `prepare-workflow`. This was a gap against
the repository's explicit remote resume/durability requirement.

Suggested repair: stage an explicit per-arm resume bundle containing checkpoint,
trace, selected state/predictions and provenance; verify its source/arm/attempt
and checkpoint cursor before restoring it. Preserve prior invocation costs and
reconcile the exact remote attempt. Do not replan a completed experiment or
silently restart it from initialization.

### P2 — Successful acceptance repeated bulk inspection three times

Owners: `experiment_workflow.py:307`, `experiment_family_workflow.py:812`,
and `experiment_family_workflow.py:696`.

The successful `accept_workflow` path inspected every arm, then the descriptor
builder inspected it again, then closure inspected it a third time. Each
inspection hashed retained artifacts and loaded predictions, selected model
and resume state on CPU. This multiplied disk IO and deserialization for large
checkpoints; this audit did not measure a real full-size acceptance slowdown.

Suggested optimization: share a digest-bound inspection snapshot within one
acceptance operation and retain a final unchanged-byte check before writes.
Do not remove independent semantic or scientific gates.

### P2 — Full physical-allocation cost was not composed by the new runtime

Owners: `kaggle_pair_runtime.py:104`, `k1_screen_training.py:475`,
and `gptrans_screen_workflow.py:310`.

The pair runtime retained observed T4 devices and orchestration wall time.
Family records measured their scoped invocation windows, with explicit limits
for bootstrap, queue and previous resume segments. There was no platform-level
allocated-device ledger covering the entire physical allocation, idle devices
or the interval after one arm completed before its peer.

The scoped measurements were not false full-job claims, but summing them was
insufficient to establish the full allocation cost required by `AGENTS.md`.
Such a ledger belongs to the platform runtime/reconciliation owner. An idle
device never authorizes an additional experiment.

### P2 — Integrated release efficiency lacked complete-path evidence

The orchestration tests replaced family recipe checks, release checks or
prospective publication in successful preparation fixtures. Independent source
imports and component tests supplied useful coverage, but not one unmocked
supported-family preparation-to-acceptance exercise. The registered workflow
also lacked the per-phase preparation timings already recorded by the server
`prepare-release` route.

Suggested validation: add one isolated, CPU-only mechanical lifecycle fixture
using the actual owners and explicitly synthetic evidence, plus phase timings.
Real-shard/GPU qualification and scientific authorization remain separate.

## Verification and measurements

Using the project's `.venv\Scripts\python.exe`:

- Focused workflow, server release, read-only preparation, sync regression,
  Kaggle push and reuse-skill tests: **73 passed in 38.96 seconds**.
- An actual immutable package built from the registered shared inventory
  contained **60 source files, 957,577 source bytes and a 207,240-byte archive**.
  Packaging took **7.430 seconds**; syntax/selected isolated-import checks took
  **1.471 seconds**, with **47 MolGap modules** imported and no import errors.
- That source-only probe deliberately omitted recipes and initialization
  bindings, producing four expected missing-input findings. Its timings were
  not a complete release measurement or a submission SLA; they excluded input
  uploads, scientific planning, platform POST, queue and execution.
- Repository `RML validate` passed with 68 trajectories, 69 evidence records,
  96 cost events, 269 role records and 40 traces. `check --frozen` passed.
- `git diff --check` passed. User-owned modifications and retained evidence
  were preserved; no RML rebuild was requested.

Ignored local audit artifacts, including exact reproduction results, were saved
under `platforms/_records/local_infrastructure_audit_20261002/`. The tracked
document records the findings without adding large payloads to Git.

The earlier broad compatibility result (**1166 passed, 9 skipped**) is retained
in [Desktop integration review](DESKTOP_INFRASTRUCTURE_REVIEW_20261002.md);
this audit added failure-oriented release checks instead of repeating that suite.
