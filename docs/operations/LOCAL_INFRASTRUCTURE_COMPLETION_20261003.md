# Local infrastructure completion — 2026-10-03

The user requested the gaps from the
[2026-10-02 audit](LOCAL_INFRASTRUCTURE_AUDIT_20261002.md) to be filled with
separate module ownership and a small addon extension surface. The work was
implemented on `molgap-server`, after the reviewed Desktop integration at
`6fec6402` and `5a85bd44`.
Concurrent server experiment commits advanced the shared branch to `6a6b7b53`
before completion. Their source registrations and operational evidence were
preserved; the infrastructure work did not take custody of those jobs.
Implementation commit: `1c3218c6` (`feat(experiment): decouple addons and add
durable recovery`).

## Implemented boundaries

| Audit gap or extension concern | Implemented owner and behavior |
|---|---|
| Addon-specific Spec branches and repeated dependency lists | `experiment_spec.py` declared typed configuration fields; `experiment_execution.py` declared reviewed addon modes and family dependencies. `experiment_source_inventory.py` selected only the required family/addon sources. |
| Final launch binding | `experiment_launch_config.py` checked the complete canonical schema, verified package/archive identity, every job, unique device assignment, actual recipe/init inputs and frozen prospective bytes. Release checks and the frozen bootstrap reused this gate. |
| Fresh-instance recovery | `experiment_resume.py` transported an incomplete native checkpoint prefix with hash-bound source, arm, attempt, trace and sidecars. Family owners checked optimizer/scheduler/RNG/cursor and retained certificate semantics. `experiment_workflow_resume.py` staged recovery against the exact original release and ACTIVE prospective bytes. |
| Repeated bulk deserialization during acceptance | `experiment_inspection.py` retained one immutable inspection snapshot per arm. Descriptor construction and closure rechecked bytes and metadata without loading tensors again. |
| Full observed physical-allocation cost | `experiment_allocation.py` measured all observed devices, including idle devices, and retained prior invocation segments. `experiment_retention.py` sealed and validated these compact records separately from scientific manifests. |
| Retrieval duplication | `kaggle_output_retrieval.py` shared bounded listing, streaming, hashing and atomic publication. The Kaggle wrapper retained its owning family or execution-manifest checks. |
| Complete-path evidence and local efficiency visibility | `experiment_workflow.py` checked recipes and all prospective arms before packaging/publication, recorded preparation phase timings, and reused the inspection snapshot. An isolated lifecycle fixture exercised real packaging, release, receipt, acceptance and terminal owners with synthetic evidence. |

The platform bootstrap hid CUDA from the CPU verification parent. Each worker
selected its assigned device before importing the family trainer. Runtime
retained the all-arm preflight barrier, and recovery used a separate preflight
directory so original training state remained intact.
Recovery captured the original prospective bytes and rechecked the canonical
records after export/staging. Replacing an ACTIVE record during preparation
could not create a different self-consistent recovery release.

An existing-family addon extended its model delta, configuration contract and
one `TrainingAddon` descriptor. It reused the family's recipe, preflight,
training, resume and output hooks. It required no new packager, receipt format,
terminal finalizer or platform launcher. The exact extension procedure and CLI
schemas were documented in the
[registered workflow guide](REGISTERED_EXPERIMENT_WORKFLOW.md).

## Verification

All Python commands used `.venv/Scripts/python.exe`. The selections below
overlapped and were not additive test counts.

| Selection | Observed result |
|---|---|
| Workflow, CLI, launch binding, lifecycle, snapshots, registration, allocation, retention and retrieval | 156 passed |
| Recovery orchestration, source isolation, resume transport, K1/GPTrans family owners, training hooks and frozen GPTrans model regression | 132 passed |
| Final strict launch boundary, including expected identity without a verified package | 14 passed |
| Shared RNG validation, native family recovery, recovery orchestration and concurrent path/EMA server compatibility | 49 passed |
| Preparation/acceptance and recovery after prospective snapshot repair | 56 passed |
| Final restore transport hash/race regressions | 16 passed |
| Final workflow recovery, including both ACTIVE-record replacement regressions | 10 passed |
| Broad compatibility selection across Spec/package/release/planning/terminal/runner owners | 833 passed, 9 skipped; the updated runtime regression fixture then passed in the targeted rerun |
| Desktop runtime regression selection after its fixture update | 4 passed |

The launch tests exercised the actual release and Kaggle before-POST checker
without making a network request. Wrong package/archive identity, duplicate
device assignments, missing jobs and missing format failed closed.

Recovery tests used synthetic CPU checkpoints and fake scheduler processes.
They checked atomic restoration, fresh-output refusal, changed checkpoint and
runtime identity, retention of prior allocation segments, and old training
exposure after a resumed preflight failure. Native K1 and GPTrans tests checked
that recovery copied the exact accepted certificate without a new diagnostic.
Source isolation tests used distinct temporary family owners and clean package
imports; they did not qualify a real accelerator run.

Repository RML validation and `check --frozen` passed. No RML rebuild was
performed. Diff whitespace and Python compilation checks passed. Concurrent
user changes to live-state, roadmap and research-memory files remained outside
the infrastructure commits.

## Qualification limits

This completion established local interface and mechanical behavior. It did not
run training, access protected evaluation roles, submit a remote job, change
scientific contracts or qualify GPU performance.

The registered preparation adapter covered Kaggle and the registered K1 v2
and GPTrans modes. Other server modes retained their existing owning
`prepare-release` route. `workflow-info` exposed that distinction.

Portable recovery required an original package containing recovery support,
unchanged ACTIVE prospective bytes, an incomplete checkpoint for every arm,
and a compatible worker runtime. Completed arms, mixed complete/incomplete
sets and extra pickled-input staging remained with the owning per-arm or
preparation adapter. Local preparation did not retry or submit a successor;
the workload skill still owned exact-attempt reconciliation.

The allocation ledger covered the Python bootstrap/runtime observation window
when execution reached the runtime. Queue/provisioning before Python and the
unobserved interval after a stopped parent remained explicitly missing.
Periodic atomic snapshots preserved progress while the parent was alive;
handled completion/failure sealed the execution retention manifest. Remote
durability still depended on the platform's independently retrievable output
and its existing retrieval/reconciliation owner.
