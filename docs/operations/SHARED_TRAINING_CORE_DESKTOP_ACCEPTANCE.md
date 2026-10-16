# Shared core Desktop adoption

## Provenance and scope

Desktop base `ea0c400c`; Server return `006a8d46`, core `c46f5695`, Common
Spec/IO `882ab5f1`. Selective import `042bdfdf` adopts the exact 39 mechanism
files in the returned inventory, not the 45-file intermediate inventory.
No whole Server branch, controller, campaign, research history or job is merged.
The [Server return](SHARED_TRAINING_CORE_SERVER_RETURN.md) is its dated report.
Fetch also observed Server `63481324`; only the later shared SDK fix `f6ec2ad0`
is selected. Its new experiment diagnostics and scientific records are excluded.

Desktop keeps its seed43 declarations, target identity, strict runtime
certificate binding, terminal/admission/RML policy and retained local-bond
architecture. Registries declare slots and a small local-bond `construct` policy;
they do not fork the shared training algorithm. Historical source packages,
scientific conclusions, production defaults and canonical RML are unchanged.

## Common compatibility follow-up

Review found that the returned reproducibility owner omitted Desktop's public
`safe_cpu_torch_load`, `nonfinite_tensor_keys`, vendored OGB observation and
atomic temporary-file cleanup. The follow-up restores those host-neutral APIs
in that same owner, retaining Server's observation/scoped finite-check machinery
and Desktop's same-dtype CPU scalar batching. It also retains the old Torch
matmul-feature guard. No optimizer, loss, sampler or precision setting changes.

CPU collection then exposed an EMA import cycle through the legacy trainer.
`foreach_ema.py` now owns its small base state class without a trainer dependency;
the cached EMA retains the existing Desktop default decay. The returned tuning
module also omitted Desktop's public accelerated entrypoints and expected an
incompatible legacy backend ABI. The follow-up retains the qualified accelerated
owner, resource/cache declarations and default backend. Slow opt-out flags are
rejected explicitly rather than ignored. Frozen historical packages are untouched.

Isolated packaging also found experiment-name detection in the shared runner.
It is replaced with optional `FamilyRecipe.resume_chunks=True`; the default
wire format is unchanged. Segmented mode policies retain their explicit chunk
behavior. New comparisons requesting chunks must freeze the option rather than
relying on an imported campaign's name. No Server experiment module is imported.

The topology-only projection is factored into `PackedGraphDataset.topology_only`
and reused by both loaders; trusted legacy coordinates are dropped from the
in-memory graph and slices, never rewritten on disk or used by the model.

The [resulting inventory](SHARED_TRAINING_CORE_FILES.json) has 42 files. Of the
original 39, 29 remain identical to Server `006a8d46`, nine have compatibility deltas
(`training_reproducibility`, `foreach_ema`, `gptrans_execution_hotpath`,
`gptrans_execution_tuning`, `family_training_recipe`, `family_training_runner`,
`family_training_data`, `packed_graph_dataset`, `source_dependencies`), and
`common/kaggle.py` incorporates upstream `f6ec2ad0`: explicit account credentials
remain instance-local without calling the SDK's global config loader.
Three dependencies are newly pinned: `gptrans_execution_backend.py`,
`experiment_resume.py` and `experiment_retention.py`.
Server must review/adopt the resulting snapshot
before claiming byte-identical cores; a sparse Desktop cherry-pick alone would
not replace its unpinned backend. Use the complete reviewed file inventory.
Do not recreate it as a Server/Desktop shim or copy an entire branch.

Static closure also found five dangling imports in the pre-existing legacy
`pcqm_gap_architecture.py` factory. The exact Server `006a8d46` model-definition
dependencies are retained: `pcqm_directed_spectral`, `pcqm_conjugated_state`,
`pcqm_hop_path`, `pcqm_local_statistics`, `pcqm_relative_value`. They add no
family/addon registration, execution route, training or scientific adoption.
The AST checker is not weakened to hide missing dependencies.
It reports all reachable missing modules together and still refuses to publish
any partial closure; security and confinement errors remain fail-closed.
The same audit restores the small existing `qm9_data.py` dependency of the
retained QM9 helpers. Its explicit acquisition functions are not invoked by
packaging or baseline training; this integration performs no download.
The existing shared output owner also needs `experiment_resume` and
`experiment_retention`. Their reviewed Server transport/ledger APIs are retained;
the former delegates safe checkpoint loading to `training_reproducibility`
instead of keeping a second implementation. Only registered shared-session
resume hooks are qualified here, not absent legacy-trainer resume facades.

## Acceptance boundary

**CPU_WIRING_ACCEPTED**, scoped to implementation `137f4139` and its pinned
mechanisms, not whole-repository cleanliness or native release.

| Check | Result | Seconds |
|---|---|---|
| 41-file primary regression | 1,551 passed, zero failed, 15 skipped | 704.07 |
| Final Common SDK/source/RML/docs batch | 307 passed, 10 unchanged baseline failures, zero skipped; no new failures | 174.15 |
| Comprehensive static closure / 42 mechanism pins | No missing internal module or pin mismatch | Not a speed measurement |
| `compileall -q src/molgap tests` | Passed | Not recorded |
| `git diff --check` | Passed | Not recorded |

These suites overlap; their passing counts must not be summed as unique tests.
The 15 primary skips are seven native-accelerator checks, six unavailable host
symlink cases and two unauthorized real-package/shard preflights. They are not
successful native tests. The grouped checks used project Python, `PYTHONPATH=src`,
CUDA/HIP/ROCR visibility masks `-1`, and OMP/MKL thread counts `1`.
The grouped suite covers shared source
identity, ordered addon gradients/optimizer ownership, accelerated K1 slots,
Live/Raw/Clean resume and endpoint inspection, isolated source packages,
Desktop legacy callers and platform bridge fixtures. The parent implements;
independent Luna runs checks with accelerator visibility disabled.

The ten baseline failures were already present against Desktop `ea0c400c`:
root compactness for `BRANCHES.md`, `CURRENT_STATE.md`, `ROADMAP.md`, plus
`test_repository_layout` cases `test_active_experiments_have_readme_entrypoints`,
`test_active_experiments_are_indexed_and_traceable`,
`test_live_control_documents_remain_bounded`,
`test_active_experiment_entrypoint_pointers_exist`,
`test_active_markdown_links_resolve`,
`test_active_directories_use_role_or_question_names` and
`test_active_sources_do_not_infer_repo_root_from_parent_depth`.
The affected root documents, experiment tree and layout test are unchanged by
this adoption. They remain unresolved, not waived into a whole-repository pass.

The once-only comprehensive dependency inventory covers the pinned mechanisms,
reviewed registry/source roots, K1/GPTrans adapter hooks and the IMS thin bridge.
Its [generated source graph](SHARED_TRAINING_CORE_DEPENDENCIES.json) records
external dependencies and dynamic-import qualification obligations as well as
the static closure: 109 roots, 111 source files, 392 edges, 50 external module
names, zero missing internal dependencies and zero mismatches across 42 pins.
All 12 dynamic-import obligations remain explicit registry/runtime work.
A complete static closure is not a claim that every dynamic
hook or third-party package is qualified on a new platform.

No real molecules, training, protected roles, remote services, model promotion
or RML rebuild are needed for this infrastructure adoption. Synthetic CPU replay
does not qualify cross-hardware bit equality, T4/A100/DCU speed, current cloud
credentials or a scientific improvement.

## Reproduce CPU checks

Use the project virtual environment and explicit CPU masks above. Primary tests:

```powershell
$primary = @(
  'shared_training_compatibility', 'shared_runtime_compatibility', 'shared_family_lifecycle',
  'common_api', 'common_contract', 'model_substrate', 'family_training_state', 'fast_sandbox',
  'source_dependencies', 'family_training_core', 'family_training_data', 'family_training_recipe',
  'family_training_kernel', 'family_training_loaders', 'family_training_session', 'family_training_runner',
  'family_full_lifecycle', 'family_execution_profile', 'checkpoint_finite_efficiency',
  'k1_execution_reproducibility', 'k1_screen_lifecycle', 'k1_two_pass_mean', 'k1_accelerated_execution',
  'gptrans_execution_hotpath', 'gptrans_execution_tuning', 'pcqm_gptrans_v4', 'graph_screen_training',
  'shared_graph_lifecycle', 'experiment_execution', 'experiment_spec', 'experiment_family_workflow',
  'experiment_package', 'experiment_preflight', 'experiment_workflow', 'experiment_cli',
  'experiment_launch', 'experiment_prospective', 'experiment_terminal', 'ims_v4_submit',
  'checkpoint_finite_validation', 'shared_resume_transport'
)
& $python -m pytest -q -ra --tb=long ($primary | ForEach-Object { "tests/test_$_.py" })
$final = @(
  'common_api', 'common_contract', 'shared_training_compatibility', 'source_dependencies',
  'shared_resume_transport', 'documentation_navigation', 'repository_layout', 'training_logging',
  'ims_v4_submit', 'research_memory'
)
& $python -m pytest -q ($final | ForEach-Object { "tests/test_$_.py" })
& $python -m compileall -q src/molgap tests
git diff --check
```

Primary completion preceded the upstream SDK follow-up; the final batch verifies
that follow-up, all 42 resulting pins and the comprehensive dependency inventory.
No implementation changed after that final inventory and batch.

Post-integration checks on Desktop `caf69ce7` also passed: the read-only
`python -m molgap.research_memory --repo-root D:/文档/molgap check --frozen --portable`
reported current derived outputs and complete portable evidence closure. No
canonical records or derived indexes were rebuilt. Final documentation navigation
had 63 passed and four root-budget cases explicitly deselected; the ten baseline
failures above remain recorded. The four pre-existing untracked Desktop paths
were preserved. This paragraph is a documentation-only follow-up to that check.

## Reuse entry

Use the [workflow](EXPERIMENT_WORKFLOW.md), [examples](FAMILY_TRAINING_EXAMPLES.md)
and [core owners](FAMILY_TRAINING_CORE.md). New addons reuse one accelerated
kernel/session and state owner. Platform skills own submission/reconciliation.
The IMS thin bridge is local-wiring support only: prepared-sidecar transport,
PBS execution and native qualification remain explicit release work.
