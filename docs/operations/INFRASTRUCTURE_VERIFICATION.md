# Infrastructure verification

This bounded record describes the reusable local infrastructure and its
verification scope. Update this record after infrastructure changes; keep
experiment outcomes in their owning decisions and route them through the
[experiment indexes](../../experiments/README.md).

## Reusable lifecycle

The [quickstart](EXPERIMENT_QUICKSTART.md) is the entry for a new agent. The
registered workflow reuses typed Spec validation, explicit source packaging,
prospective binding, immutable launch preparation, platform receipts, native
training/output inspection, portable incomplete-run recovery, acceptance, and
terminal/RML closure. [Platform adapters](../../platforms/README.md) own remote
submission, reconciliation, and retrieval; the shared CLI has no submitter.

For a new pure-2D PCQM direct-Gap family, implement one small model factory and
add the two static family/adapter registrations. A compatible addon implements
one small model hook, typed configuration, and its registry entries. Accepted
input loading, optimizer/schedule, selection, trace/checkpoint persistence,
recovery transport, output inspection, and terminal closure remain shared.
The exact model, addon, recipe, and source interfaces are in the
[graph extension contract](GRAPH_SCREEN_EXTENSION.md).
The [four-layer runtime guide](MODULAR_EXPERIMENT_RUNTIME.md) owns standard
K1/GPTrans addon slots, native Live/Raw/Clean actions, the memory sandbox and
AST source closure. New hooks compose before optimizer construction; retained
scientific recipes are not rewritten to enable a new hook or calibration policy.

Changing targets, geometry, data identity, precision, optimizer, selection, or
platform requires the owning contract/adapter and its qualification gates.
Static registration does not grant scientific comparability or compute
authority. The [screening policy](../../experiments/SCREENING_POLICY.md) owns
comparison and seed budgets.

## Local evidence

The 2026-10-14 audit exercised substrate composition, addon ownership/gradients,
native Clean-BN persistence and stale-state rejection, source closure/immutable
publication, shared training, recovery and incomplete campaign acceptance.
The final focused regression passed **472 tests, 1 skipped**.
Documentation/layout coverage passed all 35 checks, including the repaired
entrypoint/CLI guards. Active navigation and entrypoint budgets had zero issues.
Source tests publish a discovered closure and execute its function-local addon
dependencies in an isolated Python process, including package startup and
Python's package-over-module resolution. Unresolved dynamic/wildcard imports
remain explicit review obligations.
The subsequent [topology port review](TOPOLOGY_REUSE_SERVER_REVIEW.md) owns
the expanded operation-equivalence and frozen-source compatibility results.

The new-family lifecycle fixture uses a small previously unregistered model,
a simple addon, and synthetic immutable PCQM topology/role assets. It performs
actual CPU optimization through the shared trainer, prepares a source package,
checks release binding, interrupts after an epoch, transports/restores the
checkpoint, completes training, inspects outputs, accepts the retained run,
and completes terminal/RML closure in an isolated temporary repository.
Continuous and resumed model, optimizer, scheduler, RNG, and progress states
must match exactly. Fixtures do not submit jobs or reopen completed experiments.

CPU fixtures establish local executable behavior. Actual accepted-data scale,
GPU/runtime calibration, remote resource allocation, scientific acceptance,
and production promotion retain their separate owning gates.
Both actual family sandbox CLI commands passed finite forward, backward and
exact tensor/evaluation roundtrip checks. Cold process times were 6.92 seconds
(K1) and 6.72 seconds (GPTrans); warm API times were about 0.35–0.37 seconds.
These two-graph CPU checks do not certify BS128 VRAM or permanent gradient flow.

## Repeatable checks

Run from the repository root using its virtual environment:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m pytest -q --noconftest tests/test_shared_family_registration.py tests/test_graph_screen_training.py tests/test_shared_graph_lifecycle.py
& $python -m pytest -q --noconftest tests/test_model_substrate.py tests/test_family_training_state.py tests/test_fast_sandbox.py tests/test_source_dependencies.py tests/test_modular_experiment_cli.py tests/test_modular_training_integration.py tests/test_gptrans_campaign_segments.py tests/test_family_pair_acceptance.py
& $python -m pytest -q tests/test_documentation_check.py tests/test_repository_layout.py
& $python -m molgap.documentation_check --repo-root . --check-entrypoint-budgets
& $python -m molgap.research_memory --repo-root . validate
& $python -m molgap.research_memory --repo-root . check --frozen
```

The documentation command audits tracked active Markdown: local file pointers,
anchors, code pointers, and fixed entrypoint line budgets. During development,
add repeatable `--document` arguments to audit newly created files explicitly;
when supplied, they select the documents rather than extend the tracked set.
Frozen/generated trees are excluded by default; use `--include-frozen` only
for an intentional historical audit. This is a read-only navigation check.

The extended historical audit found 98 old checkout paths in frozen archive,
closed-experiment, and platform snapshots. These are not valid current paths.
Read their recorded source commit or [archive routing](../archive/README.md)
when reproducing history; retain the snapshot bytes and do not treat them as
active infrastructure instructions.

The [module index](ARCHITECTURE_MODULE_INDEX.md) routes implementation ownership.
Frequently read entries stay bounded; conditional history/directory indexes
hold growing navigation, while experiment decisions retain evidence.
Update links and focused checks when adding a module. Do not copy metrics,
job logs, or per-experiment lifecycle reports into this document.
