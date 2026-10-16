# Shared training core: server return

This dated return describes Server `006a8d46` and its 39-file snapshot. The
[Desktop adoption record](SHARED_TRAINING_CORE_DESKTOP_ACCEPTANCE.md) records
subsequent compatibility repairs and the expanded mechanism inventory.

Snapshot: 2026-10-16. Branch: `molgap-server`. Reviewed code HEAD:
`882ab5f160aa267eecafb88e7f03e5b4152a2b0f`. The documentation commit containing
this report follows that code HEAD. The worktree retained unrelated pre-existing
research README/test edits, derived records and local platform storage; they were
excluded from these commits. No whole-branch Desktop merge was performed.

## Reviewed provenance and commits

The input was Desktop `6543b38c175032ccd80d35e18b6f4fbc9ebee60b` and its
[handoff](SHARED_TRAINING_CORE_SERVER_HANDOFF.md), against Server `21a76c11`.

| Commit | Scope / Desktop review |
|---|---|
| `53b45498` | Exact cherry-pick of Common API `0edc7bdb`; Desktop already owns the original. |
| `c46f5695` | Shared lifecycle, ordered addon slots, state/checkpoint actions, SDK callers, compatibility inventory and CPU tests. Server's existing model policies are preserved in a separate policy owner. |
| `3a7127f1` | Selective import of Desktop IMS adapter `e22a0766`, plus the thin shared-worker device bridge. Desktop already owns the adapter; review the bridge delta. |
| `c5175d7b` | Server GPTrans diagnostic adapter: shared bounded loop/observers and neutral platform phases. Experiment policy/finalization stay outside the common lifecycle. |
| `882ab5f1` | Later Desktop `79f6a96a` / server-compatible `0ea5cf18` parser/IO extraction, reviewed after Desktop advanced to `ea0c400c`; retain the server catalog and caller policies. |

`d969e256` supplied shared tests and caller patterns; server declaration defaults,
strict worker admission, source/role/cost checks and private-403 absence handling
were retained. No running/frozen payload, experiment decision, production model,
canonical research evidence, AGENTS or V5 authority contract was rewritten.

## Reusable API and caller map

| API owner | Caller / responsibility |
|---|---|
| `common.declarations`, `common.identity` | Server comparison profile and prospective compatibility export; pure declarations and strict identity check. Legacy server prospective commit rebinding remains its explicit owner policy. |
| `common.kaggle`, `common.transport` | Submission/retrieval use explicit-account SDK clients, separate status/metadata reads, bounded authenticated owner absence checks, pinned no-clobber downloads. |
| `common.runtime` | Aliases original substrate/state/sandbox and family recipe/kernel/session types; checkpoint class paths remain stable. |
| `family_model_ops`, `shared_model_adapter` | Leaf model/input/loss helpers and ordered source-pinned addon hooks; seal before optimizer construction and verify parameter ownership. |
| `family_training_*` | One update, evaluation, state, schedule, selection, resume and output lifecycle. `FamilyModePolicy` carries explicit differences through the existing addon registry. |
| `family_execution_profile`, `experiment_execution` | Worker-local update/context observation and neutral phase descriptors; no global Torch clipping/backward replacement in shared sessions. |

The [core API](FAMILY_TRAINING_CORE.md), [examples](FAMILY_TRAINING_EXAMPLES.md),
[profiler](FAMILY_EXECUTION_PROFILE.md) and [Common API](../../src/molgap/common/README.md)
own usage. This report does not duplicate their manuals.

## Capability and limits

| Capability | Executed evidence / remaining boundary |
|---|---|
| Both accelerated families and ordered slots | Synthetic real-family forwards reach pre-message, post-aggregation and readout; gradients reach addon parameters, optimizer moments and serialization. K1 cached providers are included. |
| Live / Raw EMA / Clean EMA | Shared state owner; EMA updates once. Clean calibration is explicit, train-only and RNG-preserving, and persists in the session envelope. Baseline GPTrans-T has no BN and rejects Clean-BN before optimizer construction. |
| Same-runtime checkpoint continuation | CPU mid-pass continuation matches model/BN/EMA, optimizer, scheduler, RNG, exposure, partial metrics and selection. Corrupt identities/buffers reject without partial mutation. |
| 100K / 500K / full policy | Both families use one recipe/session with synthetic scale policies. No real scale, dataset or full-training qualification was performed. |
| Source package | Actual shared closure is packaged, extracted and imported in isolation; imported MolGap files must originate from that package. Selected policies are included by source inventory. |
| Kaggle | SDK, submission journal, private absence, retrieval and profiler mechanics passed CPU fixtures. No authenticated operation or new job was executed for this refactor. |
| IMS | Existing launcher validation and device-to-shared-worker bridge passed fixtures. Git-archive submission does not transport arbitrary neutral sidecars. Prepared-input transport, PBS visibility/execution, receipt mapping and native runtime qualification remain unverified. |
| Cross-owner / cross-runtime | Same source inventory and compatibility suite are supplied. Desktop selective application/review remains pending. T4/A100 speed, VRAM, precision and cross-GPU/Torch bit-exact continuation are not established. |

## Checkpoint and Desktop integration notes

Recipes without new optional addon/Clean fields retain their serialized schema.
Clean metadata and buffers live in `cursor.weight_state`; Live/Raw weights are
not duplicated. Addons must be reconstructed in declared order before restore.
Historical checkpoints/packages stay with their matching immutable source; no
source identity or missing optimizer metadata is synthesized to migrate them.

The [fixed mechanism inventory](SHARED_TRAINING_CORE_FILES.json) contains hashes
of 39 normalized source files. Desktop must install the same reviewed mechanism
contents and pass `test_shared_training_compatibility.py`; branch names are not
compatibility evidence. Catalogs, source inventories and historical receipt/terminal/admission facades
may differ explicitly outside that inventory; their moved mechanisms delegate
to the pinned common modules. Preserve server-only terminal custody restrictions. In particular, Desktop should bind its reviewed degree-bond factory
through `FamilyModePolicy.construct`, retaining its own source/objective and
selection policy, rather than importing server campaign modes or controllers.
Resolve registry/caller conflicts without forking pinned mechanism files.
The Desktop snapshot lacked the K1 provider/qualification helpers from reviewed
Server `150c6b46`, plus `pcqm_fixed_records.py` and `foreach_ema.py`. Import these
existing shared dependency files from the reviewed server source, together with
the new leaf/policy/addon owners; the inventory pins all of them. Preserve the
Desktop source-inventory/registry glue and add the shared roots and policy-hook
handling, rather than copying server experiment inventories wholesale.

## Executed checks

Environment: Python 3.11.15, Torch 2.7.1+cpu, PyG 2.6.1, Kaggle SDK 2.2.4.
All model/data execution used synthetic CPU fixtures; no real molecular role,
credential, cloud allocation or training submission was consumed.

Final common/lifecycle suite: **469 passed, 0 failed, 0 skipped**, 81.93 s:

```powershell
.venv\Scripts\python.exe -m pytest -q --noconftest `
  tests/test_shared_training_compatibility.py tests/test_common_api.py `
  tests/test_model_substrate.py tests/test_family_training_state.py `
  tests/test_fast_sandbox.py tests/test_source_dependencies.py `
  tests/test_family_training_core.py tests/test_family_training_data.py `
  tests/test_family_training_recipe.py tests/test_family_training_kernel.py `
  tests/test_family_training_loaders.py tests/test_family_training_session.py `
  tests/test_family_training_runner.py tests/test_family_full_lifecycle.py `
  tests/test_family_execution_profile.py tests/test_gptrans_execution_diagnostic.py `
  tests/test_gptrans_execution_refine.py tests/test_kaggle_submission.py `
  tests/test_ims_v4_submit.py tests/test_execution_comparison.py `
  tests/test_k1_infrastructure_comparison.py tests/test_gptrans_infrastructure_comparison.py
```

Legacy/transport regression: **219 passed, 0 failed, 0 skipped**, 52.82 s.
This group overlaps the suite above; counts are not a unique-test total:

```powershell
.venv\Scripts\python.exe -m pytest -q --noconftest `
  tests/test_k1_legacy_execution_profile.py tests/test_gptrans_legacy_execution_profile.py `
  tests/test_gptrans_execution_hotpath.py tests/test_gptrans_execution_backend.py `
  tests/test_gptrans_execution_refine.py tests/test_k1_execution_qualification.py `
  tests/test_family_training_loaders.py tests/test_family_training_data.py `
  tests/test_kaggle_pinned_exact_files.py tests/test_kaggle_family_retrieval.py `
  tests/test_kaggle_execution_retention_retrieval.py tests/test_kaggle_exact_metadata.py `
  tests/test_source_dependencies.py tests/test_training_logging.py
```

Compile and scoped `git diff --check` passed. Read-only
`python -m molgap.research_memory validate` passed; no RML rebuild/closure was
performed. Native checks and Desktop application remain the separate gates above.

Documentation/layout/reuse checks passed **39 tests**. The final pointer audit
scanned 913 documents and 3,624 pointers with zero issues; `CURRENT_STATE.md`
stayed at 120 lines. Commands: `python -m molgap.documentation_check
--check-entrypoint-budgets` and `python -m pytest -q --noconftest
 tests/test_repository_layout.py tests/test_documentation_check.py
 tests/test_experiment_reuse_skill.py`, both through the repository Python.

The late Common Spec/IO integration passed **906 tests, 0 failures, 8 skips**
in 171.86 s. This overlaps the earlier suites; it is not an additive total:

```powershell
.venv\Scripts\python.exe -m pytest -q --noconftest `
  tests/test_common_contract.py tests/test_common_api.py tests/test_shared_training_compatibility.py `
  tests/test_experiment_spec.py tests/test_experiment_launch.py tests/test_experiment_package.py `
  tests/test_experiment_terminal.py tests/test_research_memory_plan_batch.py `
  tests/test_source_dependencies.py tests/test_family_training_core.py `
  tests/test_family_training_runner.py tests/test_family_full_lifecycle.py `
  tests/test_training_logging.py tests/test_experiment_preflight.py `
  tests/test_experiment_prospective.py tests/test_kaggle_submission.py
```

The [incoming acceptance record](COMMON_CONTRACT_ACCEPTANCE.md) retains historical
hash findings on the older server-compatible snapshot. This integration imports
no historical receipts/derived state, does not diagnose those findings as facts
about this checkout, and claims no complete historical-bank portability acceptance.
