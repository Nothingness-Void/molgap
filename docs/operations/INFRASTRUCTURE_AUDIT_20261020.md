# Local infrastructure audit — 2026-10-20

The review followed declaration, packaging, launch, execution, inspection and
recovery through their existing owners. It used source inspection and isolated
CPU fixtures. No experiment outputs were downloaded; no remote job, real-data
training, checkpoint inference or scientific-contract change was performed.

## Reproduced defects and repairs

| Boundary | Reproduced failure | Repair |
|---|---|---|
| Execution failure retention | State/ledger/log IO replaced an earlier worker-spawn exception. | Independent failure receipts were attempted while the original exception propagated. Receipt failure on a successful execution still failed closed. |
| Preparation and recovery | Final report IO replaced an earlier staging exception. | Both preparation owners shared exception-aware report publication. |
| Submission uncertainty | Failure to persist `SUBMIT_UNKNOWN` replaced the underlying transport exception. | The original transport exception propagated; the pre-POST journal remained the reconciliation authority. No retry was added. |
| Completed output inspection | A completed epoch accepted `cursor.next_batch=99`. | Completed output required `next_batch=0`, matching its declared terminal exposure. |
| Resume transport | Primary artifacts with the same restore destination passed validation and failed only at restoration. | Primary source/destination uniqueness was checked before transport acceptance. |
| Resume identity sidecars | Unrecognized context and unbound cost metadata could be skipped. | Provided context/cost records required explicit matching identity. |
| Clean-EMA loading | A calibrated state loaded as ready without a caller-supplied calibration policy. | Loading Clean state required its explicit matching policy before mutation. Live/Raw-only loading retained its interface. |
| SSMA resume metadata | The SSMA exception permitted missing backbone optimizer moments. | Only the registered conditional addon parameter tail could omit moments; backbone moments remained mandatory. |
| Source confinement | An IO-module exception admitted the synthetic filename payload.py inside `src/molgap/data/`. | Reserved payload filenames inside IO directories were rejected; reviewed IO modules and ordinary `*_payload.py` source names remained usable. |
| Optional dependency closure | `import molgap as m; m.Symbol` omitted the module imported by package `__getattr__`. | Static import aliases and attribute accesses selected lazy exports through the existing AST owner; importing a package alone did not select its lazy API. |
| Shared source inventory | Recorded hashes lagged already-committed shared modules. | The same inventory member set was refreshed to normalized source hashes; Desktop application and native qualification flags were unchanged. |

The fixes tightened validation and failure reporting. Model definitions,
forward/backward math, optimizer recipes, batch sizes, data identities and
frozen executable packages were unchanged.

## Explicit capability limit

`prepare-resume` produced a local, hash-bound recovery package. The journaled
first-submission adapter did not implement continuation of an existing physical
kernel/source version. Its fixed upload allowlist had previously rejected a
resume tree with an ambiguous missing/unbound-files message. The adapter was
changed to reject such input explicitly before authentication or POST, directing
the caller to the owning platform continuation adapter after reconciliation.
No upload restriction, task identity or duplicate-submission gate was relaxed.
This review did not establish a universal one-command resumed submission path.

## Verification and operational limits

Configured Luna workers owned independent launch, package and recovery test
scopes and separate external scratch/report directories. Main-agent work was
limited to inspection, implementation and integration. The focused regressions
lived in:

- `tests/test_infra_failure_publication.py`
- `tests/test_infra_launch_audit.py`
- `tests/test_infra_package_audit.py`
- `tests/test_infra_recovery_audit.py`

Existing suites covered the corresponding lifecycle owners. Counts were
reported per scope because their dependencies and fixtures overlapped; passing
synthetic checks did not qualify a new CUDA runtime or a scientific result.

| Scope | Focused audit | Existing regression suite |
|---|---|---|
| Launch, publication and submission | 8 passed | 179 passed, 1 skipped |
| Packaging and dependency closure | 4 passed | 160 passed, 1 skipped |
| Recovery, terminal inspection and training state | 8 passed | 180 passed |

Final shared-inventory compatibility checks passed 4 tests, documentation
navigation passed 9, and skill/staging/workflow routing passed 26. Twelve
pre-existing inventory hashes were stale, in addition to the six entries
changed by this repair; the inventory's paths and qualification flags were preserved.

Recovery emitted CPU pin-memory warnings; these were not failures. The SSMA
positive case checked resume metadata with a synthetic appended addon tail;
it did not qualify a real SSMA execution or a new accelerator runtime.

Three existing experiment-document pointers were repaired without changing
their historical evidence. An example synthetic package path in this report
was clarified as a filename rather than a repository navigation target.

Indexed loading already retained file-identity verification separately from
its bounded mmap cache; the earlier repeated-whole-shard-hash observation was
not attributed to this implementation. Package interruption still left partial
local preparation for reconciliation rather than silently overwriting it.
Existing user edits, derived RML views and active frozen task custody were preserved.

## Desktop simplification integration

The subsequent authorized synchronization selectively applied Desktop commits
`711d2c87`, `1c3003c4`, `22deb5c6`, and `3eead61d` to the Server integration branch.
It reused exact-byte package verification within one preparation/lookup,
unified repository-relative initialization, collected RML records in one walk,
deduplicated validate-only source rechecks, and added scoped local tests,
portable host test bundles and bounded retired-cache cleanup.

Server fault-publication and recovery repairs were retained. Mount checks still
included every staged package sidecar. Modern2D CPU defaults and explicit native
admission were preserved; mature-family metadata followed the shared catalog.
The RML collector retained Server's Unified-V2 input-role classifier. The new
test/cleanup tools were explicitly excluded from training runtime distributions.
The [quickstart](EXPERIMENT_QUICKSTART.md#6-check-affected-flows-and-manage-temporary-storage)
owned usage instructions. No remote submission or scientific computation followed.

The integration was committed as `7a7c4388`. Post-integration local validation
was delegated to three independent configured Luna workers:

| Scope | Final result |
|---|---|
| Packaging, preflight, receipts, lookup, preparation and earlier fault regressions | 301 passed, 2 skipped |
| Local runtime, storage, scoped testing and family catalog | 39 passed |
| RML discovery, planning, role classification and recovery | 60 passed, 6 skipped |
| Shared inventory, documentation navigation and reuse routing | 17 passed |
| Isolated worker distribution / Git-independent test fixture | 1 passed in checkout and 1 passed from the extracted test package |

The last row was one test rerun in two environments, not two unique cases.
The Windows RML symlink checks lacked native privileges. The distribution check
retained actual synthetic CPU K1/GPTrans addon, Session and state assertions.
Its fixture setup was adapted from Desktop so an extracted test package did not
require Git metadata in its own root; temporary fixtures still froze source in Git.
The portable test inventory omitted a Desktop-only policy input and selected the
existing Server bootstrap regression. The host credential compatibility alias
was also excluded from worker runtime. All final affected checks passed; live
submission, full-data performance and new native qualification were not evaluated.
