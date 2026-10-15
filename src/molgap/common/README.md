# Common API

One mechanism for desktop and server, within this repository. No separate
service, duplicate trainer, controller import or cross-machine coordination.

Start with the [workflow](../../../docs/operations/EXPERIMENT_WORKFLOW.md).
This API factors mechanisms out of the existing owners; it grants no execution,
role, submission, promotion or full-admission authority.

## Public surface

| Module | API | Boundary |
|---|---|---|
| `declarations` | `ExecutionQuestion`, `ExecutionProfile`, `build_execution_proposal` | Pure translation into existing prospective RML inputs; no publication or authorization. |
| `declarations` | `safe_comparison_base`, `run_version`, `comparison_base(..., arm_ids=...)` | Portable storage names with explicit question/arm identities. |
| `identity` | `validate_execution_identity` | Strict source/run/action/attempt check; never rewrites provenance. |
| `kaggle` | `api_for_credentials`, `dataset_status`, `dataset_details`, `kernel_source` | Explicit-account SDK compatibility. Status is not metadata or proof of native execution. |
| `transport` | `download_pinned_files` | Exact allowlist, SHA256, atomic no-clobber retention; no bulk fallback or scientific acceptance. |
| `runtime` | Existing substrate, state, sandbox, graph-model and dependency APIs | Object aliases, not replacement types, loops or registries. |

Import the submodule required, not all runtime dependencies at startup:

```python
from molgap.common.declarations import (
    ExecutionProfile, ExecutionQuestion, build_execution_proposal,
)

profile = ExecutionProfile(
    owner="desktop", track="B", platform="kaggle1",
    hardware="Tesla_T4x2_shared_allocation",
    cheapest_falsifier=contract["cheapest_falsifier"],
    estimated_device_hours=contract["estimated_device_hours"],
    estimated_wall_hours=contract["estimated_wall_hours"],
)
proposal = build_execution_proposal(profile=profile, **question_arguments)
```

The caller supplies all question arguments: question, arm, trajectory/source/run
identity, version, contract/authority/role/decision references and timestamp.
A proposal is a draft. Existing RML schema/policy validators remain authoritative.
`None` cost is measurement-missing, never measured zero. Estimates stay estimates.
No owner, account, seed, hardware, dataset or budget is inferred from the host.

## Existing infrastructure

`runtime` exposes the existing `ModelSubstrate`, `bind_substrate`, slot payloads,
`CalibrationPolicy`, `FamilyTrainingState`, `build_graph_model`,
`run_sandbox`, `run_arm_sandbox`, `discover_source_dependencies` and
`registered_dependencies`. Original module paths/classes stay intact for
checkpoint compatibility. Use the [four-layer runtime](../../../docs/operations/MODULAR_EXPERIMENT_RUNTIME.md)
for call contracts. Training still uses the existing
[accelerated family core](../../../docs/operations/FAMILY_TRAINING_CORE.md);
platform skills own remote operations. This package does not wrap another
training launcher around that core.

## Compatibility and limits

`execution_comparison.py` keeps legacy T4 question defaults as an explicit
compatibility profile and delegates declaration construction here.
`kaggle_submission.py` and output retrievers call the shared SDK/transport.
Desktop strict prospective validation calls the shared identity check.
Keep frozen packages and historical JSON unchanged.

Declaration/identity modules and package initialization are in the reviewed
worker source inventory. Host-side credential/transport operations are not
forced into training payloads; source-package safety exclusions remain intact.

Acceptance-plan formats, experimental recipes/addons, controller ownership and
full-admission rules remain with their current owners. In particular, the server
legacy prospective commit-binding policy is not silently changed by this
extraction; it is not the strict common identity API. Sharing a module does not
make differing historical acceptance schemas equivalent. Further convergence
must adapt these explicit contracts, not merge an entire machine branch.

For a change to these modules, use one reviewed common patch in both owner
branches; adapters may differ, the common files must not. Run
`tests/test_common_api.py` plus the affected adapter suites on both snapshots.
The common suite checks pure declarations for both owners and several platforms,
legacy forwarding, provenance rejection, SDK call shape, no-global-auth behavior,
pinned no-clobber downloads and actual static package closure. It uses synthetic
CPU/files and fake APIs only, not native runtime or scientific qualification.
