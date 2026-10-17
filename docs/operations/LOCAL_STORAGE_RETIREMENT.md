# Local storage lifecycle

BRANCHES.md owns branch routing. Experiment contracts/decisions and remote durable
artifacts own scientific facts. A physical checkout is temporary execution space;
its removal does not close science, adopt a model, cancel a job or release training.

## Keep and retrieve by question

Keep canonical RML/decision metadata and the necessary selected/replay weights,
source/configuration identities, hashes and durable remote locators for the question
being used. Preserve required unique predictions, traces and resumable states under
the applicable contract. Never discard the sole copy of private source or evidence.

Do not hydrate an inherited experiment/RML catalog. Before downloading, name an
explicit artifact allowlist and estimated total size, check existing verified
custody, then retrieve only those artifacts. Unknown size is unknown, not zero.
Reproducible caches, bytecode, dependency/build bundles and transport copies are
temporary; retain their identity/rebuild or retrieval instructions instead of all
copies. Keep one verified necessary artifact copy; a hardlink alias is not another
independent backup or another allocation of logical bytes.

## Retire physical checkouts

1. Identify the exact owner, HEAD/ref, source/configuration and scientific status.
   Check active local process/editor dependencies before removing its path.
2. Preserve exact staged, modified, deleted and untracked source/RML state in a
   recoverable snapshot. Inspect ignored files for unique source/evidence/artifacts.
   Retain immutable source and verified recovery/archive refs before deleting refs.
3. Hash and verify required unique artifacts in durable custody; record manifest,
   remote identity and retrieval instructions. Preserve unresolved qualifications
   and unknown remote state without rewriting canonical evidence.
4. Verify recovery and the resolved target path, then remove the worktree and its
   disposable local payloads. Skip reparse/junction traversal; remove links only.
   Reuse the primary .venv; never delete its target through another checkout link.
5. Keep the primary checkout and worktrees needed for active local work. Completed
   work retires promptly; pending science may also use cold snapshots. Positive
   unadopted work stays unadopted. Restore only the owner needed for later work.

## Availability and retirement index

Record eviction and actual availability in a separate local overlay linked to
verified custody/retrieval receipts. Do not change frozen artifact hashes or
scientific records to disguise absent local payloads. Existing portable validators
remain unchanged; report missing artifacts and check scope honestly. Generated RML
indexes/completeness reports are navigation, not primary scientific evidence.

The [retirement index](LOCAL_RETIREMENT_INDEX.json) is the portable lookup for
owner ID, original HEAD, verified archive/snapshot ref, scientific status, physical
disposition, repository-relative canonical path and external custody-manifest SHA256.
It is generated from verified retirement receipts; only a published, verified
entry establishes snapshot custody. It carries no payloads or new scientific
schema/runtime. Restore from its exact ref and custody manifest, not a retired
absolute directory name.
