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

## Inspect disposable preparation caches

Use the configured interpreter and the existing checkout wrapper:

```powershell
& $python platforms/local/run.py --profile $profilePath --module molgap.local_storage -- --profile $profilePath --repo-root .
```

This bounded listing reads direct scratch and runtime-cache entries without
opening payloads. To remove a reported `DISPOSABLE_RETIRED_CACHE`, add its exact
`--candidate` path and `--execute`. Execution rechecks the selection and removes
only the generated files in an atomically retired `.incomplete-*` cache directory.
An ineligible selection blocks the batch before deletion begins.

Normal invocation scratch still cleans itself on exit. Scratch of unknown
ownership, package locks/staging directories and complete caches remain report-only;
age alone is not proof that they are unused.
This command does not retire worktrees or remove scientific records. Interrupted
removal reports `STOPPED`; inspect the remaining directory before retrying.

Successful preparation retires earlier unpublished failed preparations only when
their payload bytes match the verified replacement. Unique files, uncertain
publication, operation journals and links block retirement. Compact reports and
an inventory receipt survive; interrupted deletion resumes against that receipt.
After a Kaggle submission reaches `BOUND`, its journaled cleanup removes verified
upload payload copies and `source_roundtrip`. Reentry uses the bound identity and
retirement receipt; failed or unknown publication keeps its recovery inputs.
Explicit resume preparation uses a temporary input view from the retained frozen
source package and original initialization pointers, checked against the upload
receipt. The view is removed on success or failure. Missing or changed originals
stop recovery; cleanup does not authorize downloads or regenerated initialization.

Recipe metadata may set `resume_keep_count` to retire older resume packages that
are exact duplicates of retained packages. Omission retains existing behavior.
The count never authorizes deleting a unique resumable state or evidence file.
These paths do not sweep legacy complete caches, historical evidence or active worktrees.

New source packaging uses `runtime-cache/bounded-v1`, isolated from legacy
builders. A nonblocking operating-system lock covers cache reads, publication and
eviction; busy caches fall back to direct packaging. Each run keeps an independent
source package. The cache normally retains eight recognized completed entries;
`MOLGAP_SOURCE_CACHE_MAX_ENTRIES` selects 0 through 128 (0 bypasses caching).
Only recognized generated cache entries are eligible, and cleanup failure cannot
invalidate the run's package. Old cache roots remain report-only.

New resume chunks may share unchanged payload files with the preceding verified
immutable bundle of the same owner and attempt. Mutable training outputs are
always copied. Unsupported hardlinks fall back to verified copies; shared names
are not independent backups. Unique checkpoints and complete curve history remain
retained. Partial or unknown old chunks are reported and preserved, and optional
retention cleanup cannot fail an already committed checkpoint transaction.

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
