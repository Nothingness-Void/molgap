# Research workflow reform — cumulative migration record

Date: 2026-10-16. Implementation baseline: server
`1179054750fd8ff500548ad6b3ee3eece94bb786`.
Input: `MOLGAP_REFORM_PLAN.md`, SHA256
`e617a4f32da311942d48a8ca1b11a26ca125c351509410aaba72e321d85a174d`.

## Acceptance scope

Status: **INFRA_CPU_ACCEPTED for the delivered server subset**;
the entire A–G reform is **NOT_COMPLETE**.

The bounded server implementation is CPU-qualified, not a completed deployment
of every reform batch. No remote POST, credential inspection, GPU allocation,
real-data run, graph rebuild, official/protected-role read or scientific ranking
was performed. V5, historical scientific decisions and canonical RML are unchanged.
The [quickstart](EXPERIMENT_QUICKSTART.md) owns the normal workflow; this is its
one cumulative implementation/evidence record, not another operating protocol.

| Batch | Delivered | Qualification / remainder |
|---|---|---|
| A | Default pytest collects `tests`; CLI fixtures load on demand; selected checkout runs under `-I`, including owning platform scripts | CPU accepted; shared `.pth` and running environments were not changed |
| B | Machine-local external roots, stable run/attempt paths, verified copy and bounded metadata census | New path/copy accepted; census cannot prove unique-copy custody or authorize cleanup; no bulk migration/GC |
| C | One catalog and read-only projections; shared K1/3 and GPTrans/2 scientific inputs generate recipes, Spec, workflow, source and prospective bindings | Bounded CPU scenarios accepted; arbitrary recipe/new-family compilation is not implemented |
| D | Fixed runtime distribution, bulk Git, atomic complete archive cache; V2 explicitly distinguishes generated JSON from committed source | CPU package accepted with V1 compatibility; no native qualification |
| E | Existing Kaggle `--experiment`/journal entry; original-run inspection, acceptance, recovery and pinned-retrieval adapters; retained local timings | CPU/fake-API scope only; authenticated latency and real receipts not measured |
| F | One normal guide, single portable source inventory and explicit commits | Server patch supplied; Desktop is NOT_APPLIED and requires its own review/tests |
| G | Public model/recipe/session/checkpoint paths preserved | Deferred until cross-owner integration; no mass source moves or compatibility constants deleted |

The remaining C/F/G work is real work, not declared complete by the CPU status.
Reference-input V1 remains compatible. Scientific-input V1 compiles bounded
optimizer/catalog-addon changes through existing scientific builders and gates;
it does not make arbitrary recipe/model changes executable or admissible.

## Reuse and replaced defaults

Reused `common.spec`, canonical fingerprints and IO, `FamilyTrainingCore`,
Recipe/Session/Logger/Profiler, existing fixed-data and cross-scale validators,
prospective/RML planning, source-package V1, launch/acceptance/recovery owners,
Kaggle preparation and submission journal. No replacement trainer, optimizer,
sampler, EMA implementation, RML database or platform submitter was introduced.

- Removed eager six-CLI imports from root conftest; dependent tests request their
  existing loader fixture explicitly.
- Removed automatic repository document-length assertions. Optional legacy
  budget CLI behavior remains callable, not an ordinary acceptance gate.
- Removed manually duplicated ordinary addon/family execution tables. The
  catalog supplies both parsing metadata and executable descriptors. Legacy
  `TrainingAdapter`/`TrainingAddon` APIs remain read-only projections.
- New editable preparations derive tracked runtime Python automatically; old
  selected-source/AST APIs remain for compatibility and development diagnosis.
- Commit verification now uses one tree read and one batch object read. The
  source archive cache never reuses another Spec/package/release/receipt.
- Removed the addon guide's duplicate handoff sequence; detailed documents are
  API/compatibility references to one normal quickstart.

## Local storage and retained originals

The user rejected C-drive artifact storage. The machine-local profile is
`E:/MolGap/machine-profile.json`, with roots `E:/MolGap/data`,
`E:/MolGap/artifacts`, and `E:/MolGap/scratch`. The interpreter remains the
existing project venv on C; no environment relocation was attempted. Data-root
configuration does not rewrite an accepted dataset manifest or relocate a cache.

One existing source archive was copied, stream-hashed and rechecked at:

```text
E:/MolGap/artifacts/source-packages/5bb2057ddf9f9691780f8572bfb57c50f8975dec8a7e5a733988ea169e80f57c/source.tar.gz
```

It contains 9,757 bytes and SHA256
`5bb2057ddf9f9691780f8572bfb57c50f8975dec8a7e5a733988ea169e80f57c`.
The source under the old `platforms/_records/local/modern2d_review_hardening/archives`
location was retained. No original, worktree, ignored payload or branch was deleted.
Historical manifests were not rewritten. This is a physical locator supplement,
not a new source/scientific identity.

Only registered worktree metadata was inventoried; ignored/unique-payload
censuses of all old worktrees remain unperformed. Five worktrees were registered
at observation, including the primary server, archive, two detached checkouts and
the reused reform worktree. No cleanup eligibility is inferred from that count.

## Actual CPU verification

Runtime: project Python 3.11.15, Torch 2.7.1+cpu, PyG 2.6.1; CUDA hidden and CPU
threads bounded. Raw test reports are retained under
`E:/MolGap/artifacts/reform`; pytest scratch directories are on E and retained.
Counts below overlap and must not be added as independent experiments.

| Raw JUnit name | Observed result | Scope |
|---|---|---|
| `catalog-package.xml` | 15 passed, 19.52s | Catalog equivalence, source safety/cache and A/B fixtures |
| `package-legacy-recheck.xml` | 128 passed, 1 skipped, 74.62s | Existing packaging/common/recipe compatibility; native symlink privilege unavailable |
| `entry-journal.xml` | 84 passed, 36.31s | New inputs, original fake-API journal, family output inspection |
| `distribution.xml` | 1 passed, 33.42s | Actual 426-file fixed distribution in an isolated child; existing core/state/session tests run inside it |
| `core-regression.xml` | 268 passed, 18.85s | Core, Recipe, Session, state, common API, catalog, storage, registry and input compatibility |
| `lifecycle-regression.xml` | 305 passed, 1 failed, 1 skipped, 134.24s | One stale CLI command-set assertion; no package/journal production failure |
| `gates-cli-recheck.xml` | 233 passed, 54.20s | Corrected CLI guard, new workflow, Modern2D CPU restrictions and cross-scale no-auto-decision rules |
| `final-front-door.xml` | 107 passed, 27.43s | Final run-ID locator, local CLI, checkout/platform launcher, portable inventory and fake-API entry integration |

The original pre-storage-choice runtime report was also copied to E as
`initial-runtime.xml`, with its original retained. The isolated child's raw
JUnit was copied to `isolated_cpu_state.xml`, and its observed timings/module
path to `distribution_observation.json`; each copy used the same streaming
no-clobber helper. All reports preserve failed attempts rather than replacing
them with later green results.

The isolated child's copied report contains **116 passed**. Report byte pins:

| Report | SHA256 |
|---|---|
| `catalog-package.xml` | `7100a3b59cb1bbbf625701d4ea9613be2419075d350c2e9edb99a9bb05da1a65` |
| `core-regression.xml` | `597bd76c9141a1408d2dd31a983a2fe9170b307b1dad3c9c519fa27bed882bc3` |
| `distribution.xml` | `942300e0b25d92e02b1f8d0466826297d6fc3c4ec5fdf6be8a8353e4cf183f03` |
| `entry-journal.xml` | `c060f31eaaa4e1df3cadeee4acc09593b32181e365eb40d9e1a28c825ab05a0a` |
| `final-front-door.xml` | `ec065defffe2de4880e4da24791ca3bcb88137b80bd956447fab53a7b6986edc` |
| `gates-cli-recheck.xml` | `f81406a33bf741a81adce1bddf2744e19bf35db241f4843cdc275e84ac2cf6c5` |
| `initial-runtime.xml` | `d8852c0adcc8c07d02111e6ff5009ff4bb5535053b9ef059695b8ce506bd542c` |
| `isolated_cpu_state.xml` | `97ba92585f1b043faffafbae058b46ead522ac6c0b1b201cbcaffdf5ecb0e1da` |
| `lifecycle-regression.xml` | `be9df2f9cce609c79736d867ed5252e128a34cba56b68c3ef185ed4faf2f5a42` |
| `package-legacy-recheck.xml` | `3553a77d3d9b177948f67a9fde0a471c58213d1a65f7607ce0ece121e226231d` |
| `package-legacy.xml` | `33bdaed7843859ce7b0978b813470a8cac622710eaa944e872d4d30c5098043b` |

The earlier `package-legacy.xml` retained 73 setup errors because the fresh E
scratch parent did not yet exist (56 tests passed). Creating that explicit parent
and rerunning produced `package-legacy-recheck.xml`; raw failure evidence was
not deleted. The command-set regression was updated to enumerate the new local
commands, without relaxing the no-network/no-dynamic-dispatch checks.

The isolated distribution test verified every imported MolGap module came from
the extracted E-drive source, despite a deliberately incorrect inherited
`PYTHONPATH`. It exercised actual random K1 then GPTrans single-batch addon
forward/backward/state roundtrip via the existing sandbox. It copied the existing
CPU core/session/state test drivers, not a training implementation. Continuous
and interrupted/resumed optimizer, scheduler, RNG, cursor, EMA and selection
states are checked by those owners; synthetic BN/Clean calibration has its own
rollback/roundtrip checks. Scaler is inapplicable to the retained FP32 CPU path.
Twelve scientific model/training-owner source files matched the pre-reform
baseline byte-for-byte after LF normalization.

## Measured performance, not targets claimed as results

One actual 426-file distribution archive contained 1,301,148 compressed bytes.
At source commit `f4c9afdd6fb89eaa361d8046dcf0c0670684a591`:

| Measurement | Raw seconds |
|---|---:|
| Cold source-package/cache publication | 7.4543983 |
| Warm source-package creation using the same archive | 3.8344914 |
| Independent CPU child, actual family sandboxes and state/session tests | 15.2916196 |

The cold/warm packages shared archive bytes but had different Spec/package
identities. Another 80-file fixture verified repository checking starts exactly
three Git processes, independent of source-file count. Tampering hidden by
`assume-unchanged`, cache corruption and scoped credential names are rejected.
The cache key includes builder version, exact source commit and all source
payload hashes. A changed source/record commit invalidates it conservatively.

These are single raw samples, not p50/p95 estimates. Full local preparation
against a live authorized declaration, upload/source-ready time, actual receipt
latency and queue time were not measured. The 90-second/5-minute targets are
therefore NOT_ESTABLISHED. No seed-43/44 experiment was created to test caching.

## Portable commits, rollback and owner boundaries

Each batch has a separate local commit on `codex/reform`, based on the server
baseline above:

- `35892fab`: A, lazy collection and isolated checkout launcher.
- `d4a7a51c`: B, explicit machine roots and verified streaming copy.
- `744a128a`: C, single catalog projection.
- `fe9baca0`: D, fixed inventory, bulk Git and archive cache.
- `8fd264b2`: E, reference freeze and original platform entry.
- `027164cf`, `0670dd42`: child scratch and platform-script source selection.
- `f4c9afdd`: actual isolated distribution verification.
- `a72ca779`: original preparation/journal fixtures through the new entry.
- `84f776cf`: read-only run-ID locator.

The [portable inventory](REFORM_SHARED_FILES.json) reuses the existing
`molgap-source-files-v1` format and pins 49 shared files after review hardening. The original
[historical inventory](SHARED_TRAINING_CORE_FILES.json) remains unchanged at
`f6ec2ad02ea9105c75217bd7e9e0cdb70331229d`. Catalog/platform glue must be reviewed
against Desktop's own capabilities rather than copied as machine policy.
Desktop tip `1ebb2e45d78d623374b50fb5df40b7af8e51ce59` was inspected read-only;
its kernel, loaders, recipe and session differ from server, so file-name matching
is not a portability certificate. No Desktop branch was changed.

Rollback is per implementation commit/new-entry selection. Frozen historical
runs keep their original source/dependencies/artifacts; no hot update is applied.
Submission ambiguity always uses the original journal, never a new run ID/slug.
No push or remote branch update is authorized by this reform.

## Remaining work

1. Scientific compilation is bounded to shared K1/3 and GPTrans/2, catalog
   addons, learning rate and weight decay. Other typed recipe/input/family
   changes require their existing owning builders and focused acceptance.
   Frozen screen budget, data, batch, seed, sampler and selection cannot be
   silently changed through this input. No baseline is retrained automatically.
2. Desktop must apply/review the portable mechanisms with its own catalog glue,
   preserve its unique recovery/execution semantics and pass the same inventory
   and targeted compatibility tests. Native/runtime qualification stays separate.
3. G source regrouping and historical constants migration follow that cross-owner
   acceptance; public checkpoint module paths remain intact in this delivery.
4. The bounded metadata census described in the follow-up does not establish
   historical unique-copy recoverability. Further copy-only migration remains
   separate; no deletion or generic TTL/GC service was implemented.
5. Real platform latency/receipt samples and any native test require separately
   applicable authority; this CPU task does not release them.
6. V1 archives contain frozen recipe records as well as code. Changed recipe
   bytes or commits conservatively miss the cache, even if model Python is
   unchanged. Logical Spec-only reuse is verified; seed-only reuse across
   changed recipe records is NOT_ESTABLISHED. A separate runtime/config package
   layer was not introduced. V2 adds explicit generated-record provenance, not
   removal of recipe byte binding or seed-only reuse claims.

## Primary server integration verification

After a read-only remote fetch, the primary `molgap-server` checkout was
fast-forwarded from `1179054750fd8ff500548ad6b3ee3eece94bb786` to
`5985e26b8305ac7c65ccff69d2c24b9e63557742`. All 14 unrelated dirty tracked
files matched their original SHA256 pins before and after integration, including
`tests/test_research_memory.py`. They were not staged or included in the patch.
Unrelated untracked/ignored evidence was retained, not adopted or cleaned.

The final primary suite (`primary-acceptance.xml`) passed **41 tests**, with
zero failures/errors, in 61.098s (JUnit suite timing). Its byte pin is
`2a31c213e309147d2b83072e8653e57510b3149ec4874e9a8a58cd4922236189` under the same E-drive report root. It reran
the actual committed fixed distribution and isolated family/state/session proof,
as well as input/run-ID, cache, catalog, path and platform-journal integration.
The primary distribution sample used 426 source files;
cold/warm packaging took 6.068310/3.710259s and the isolated CPU
child took 15.619401s. These remain raw single samples, not platform latency.

The 14 changed active Markdown documents passed targeted navigation audits:
317 pointers before integration and 318 after, eight anchors, zero issues,
no line-count gate. This does not claim
a clean-checkout audit of every historical/untracked evidence link. Twelve
scientific execution owners, both durable V5 contract texts and the historical
shared inventory had no diff from the implementation baseline.

No push, Desktop integration or worktree removal followed. The local reform
branch/worktree was retained for recovery; remote reachability of the new commits
is not claimed. The existing RML derived/user changes remain the user's own.

## Review hardening of 13eed22e (2026-10-16–17)

Review input SHA256:
`edc25ec00a9a15afc9d48bf57e5f589096f016c46f4ad1c59080ceb6f6176e1f`.
The supplied review was implemented without a new trainer, RML database,
platform submitter or approval system. Its core scenario, not test count alone,
is the acceptance target:

- One committed human learning-rate edit generated a new recipe, Spec,
  workflow, purpose-scoped comparison and prospective inputs using existing
  owners. K1 and GPTrans were tested independently in isolated E-drive fixture
  repositories with actual model initialization schemas and actual package,
  staging, release and RML checks. These were CPU synthetic infrastructure
  fixtures, not scientific experiments or accepted native references.
- An injected staging IO failure preceded RML publication. The same front door
  retained its complete failed report, regenerated staging with identical frozen
  input/Spec identity, published one prospective record and reused its completed
  preparation on the third call. No SDK/remote request occurred. Published,
  unknown, active PREPARING and operation-journal states remain non-retryable
  here; the original platform journal owns remote reconciliation.
- Source caches are validated in temporary directories and published by one
  directory rename. Legacy partial cache objects are retained and rebuilt from
  their bound source; corrupt complete objects still fail closed. V1 packages
  remain unchanged. V2 retains six sidecars while explicitly binding bounded
  generated JSON and the committed human input; untracked Python is forbidden.
- Root editable-install ambiguity was removed. The indexed CPU-only prototype
  hashes once per loaded shard instance across 100 repeated reads, invalidates
  changed file identities and retains its real-data prohibition. This metadata
  cache is not a malicious-filesystem or native loading certificate.
- Permanent old-source/old-capability locks were removed from ordinary tests;
  prior one-time byte proofs remain in this record. Current package hashes,
  state/gradient/recovery behavior and source isolation remain tested.
- Local errors expose finite safe field/reason messages; arbitrary exception
  strings and SDK/HTTP request details are not echoed by the front door.
- Baseline data declarations must match the Spec identity bound by the retained
  reference context. Changed membership cannot borrow an unchanged recipe and
  prediction manifest to manufacture a compatible comparison. This binds the
  historical reference input, not the candidate's platform/run identity.

| Raw JUnit name | Observed result | Scope |
|---|---|---|
| `review-repairs-2.xml` | 85 passed, 1 skipped, 78.014s | Recovery/input/cache and old package compatibility |
| `review-regression.xml` | 141 passed, 242.061s | Existing workflow, acceptance, comparison, catalog, indexed CPU and fake-journal fixtures |
| `review-final.xml` | 106 passed, 1 skipped, 101.765s | Final R1–R6, generated-package safety, V1 compatibility and original journal front door |
| `committed-review.xml` | 169 passed, 146.429s | e4a11e04: actual K1/GPTrans config/recovery and two-arm compilation; committed isolated distribution plus Core/Recipe/Session/state tests |
| `data-context.xml` | 2 passed, 14 deselected, 40.23s | eda23d16: changed baseline membership rejected in both families before publication |

These overlapping suites are not independent scientific experiments. The skip
is the existing Windows link-privilege limitation. Earlier fixture failures were
retained on E. `scientific-final.xml` had 32 passes and one instrumentation
failure: counting started after the constructor had already verified/opened the
shard. Counting before construction passed in `review-final.xml` (one hash for
construction plus 100 reads, then one additional hash on each identity change).
No failed infrastructure test was recorded as a model failure or hidden by xfail.

The committed distribution contained 426 source files (1,303,554 archive bytes).
Its cold/warm package samples were 5.434930/4.127232s; isolated CPU checking took
15.772613s. These are single local samples, not authenticated preparation/upload
latency. The actual compiler scenario used original release/RML checks with
synthetic artifacts; fake-API journal tests retained their explicit qualification
stubs. Neither proves an actual platform execution or scientific improvement.

Portable implementation commits: `e4a11e04` (compiler and R1–R6) and `eda23d16`
(baseline data/context binding). Applying these mechanisms to Desktop still needs
review of its own catalog and runtime semantics, not a whole-branch copy.

Raw review regressions and the final scenario report are stored under the
existing E-drive report root. No historical V4/V5 decision, durable scientific
contract, existing RML record or primary user change was edited. Desktop/native
application and end-to-end live platform latency remain unqualified. Overall
A–G reform completion is still NOT_COMPLETE.

### Primary acceptance (2026-10-17)

The primary server fast-forwarded from `13eed22e` to `06c3abd8`. All 14
pre-existing dirty tracked files retained their original byte pins before and
after integration; `tests/test_research_memory.py` was neither edited nor staged.
Unrelated untracked/ignored evidence and all registered worktrees were retained.

`review-primary.xml` passed **188 tests**, zero failures/errors/skips, in
165.253s (JUnit timing). SHA256:
`c25ab77f2b630f1c6a7faf1eccd94948542ea14fc118011d3b6cf22aec881d87`.
It reran both complete scientific-input scenarios, two-arm bindings, changed
reference membership rejection, R1–R6 regression cases, original fake-API journal
integration, actual committed distribution, and Core/Recipe/Session/state tests.
Each family retained one frozen binding and exactly one published prospective
record after recovery; actual platform requests/submissions were zero.

The primary source-package cold/warm samples were 5.061107/3.729428s; the isolated
CPU child took 15.950323s. Its source commit was `06c3abd8`, with 426 files and
1,303,554 archive bytes. These remain local samples, not live platform latency.
Compact acceptance metadata and raw JUnit are under
`E:/MolGap/artifacts/reform/review-primary-acceptance.json` and the report root.
Five changed entry/operating documents passed a read-only navigation check:
176 pointers, six anchors, zero issues, no length gate.

The scientific Core/Recipe/Kernel/Loaders/Session/State/Runner, durable V5 texts
and historical shared inventory had no diff from `13eed22e`. This was a bounded
**INFRA_CPU_ACCEPTED** delivery, not native qualification or overall reform
completion. No push, training submission, Desktop branch edit or cleanup was
performed in this review-hardening task.

## Frozen-run continuation follow-up (2026-10-17)

The review of `c3f4c97b` identified an input-ordering defect: a repeated
scientific input compiled against a newer HEAD before discovering its frozen
run. `c5b49249` repairs the existing resolver/preparation entry, not the trainer,
package/cache owner, RML planner or submission journal.

- New runs compile/freeze once. Existing runs first compare the canonical human
  input and validate their original frozen Spec/workflow. Completed preparations
  verify/reuse their original package without current Git, development references,
  generated definitions or original initialization paths. Changed inputs reject
  overwrite and require an explicit new scientific run. Legacy six-field
  scientific bindings remain readable and are not rewritten/migrated.
- New freezes additionally pin their source commit. Unpublished retries under
  another HEAD, missing original source bindings, partial freezes and ambiguous
  publication/platform effects do not compile replacement identities. The
  operator must supply the original source checkout for unpublished recovery;
  automatic historical checkout/restoration was not introduced. Completed
  packages retain the original journal path, including `SUBMIT_UNKNOWN`.
- Both actual family fixtures prepared on A, injected a pre-publication IO
  failure, committed unrelated documentation B, verified no substitution or
  mutation, and recovered under A. After completion another documentation commit
  and temporarily unavailable reference/generated/init paths still allowed
  reuse and freeze-input calls. A compiler sentinel proved no compilation on
  continuation; frozen/prepared bytes stayed identical and each original
  trajectory was published once. All unavailable fixture files were retained
  and restored, not user artifacts.
- Both registered readout-scale addons passed the complete scientific-input,
  recipe, comparison, real package/release/init and prospective CPU path.
  Their declaration source SHA and packaged recipe addon order agreed. The
  compiler distinguishes a verified reference/composed default-loss mode label
  from a real loss change; the owning mode-tagged Spec/recipe digest is unchanged.
  Architecture-plus-optimizer confounds remain rejected by the original comparator.

`continue-complete.xml`: **58 passed**, zero failures/errors/skips, 181.032s;
SHA256 `5aa4e1f543a61622a31354976f152d32e0bf7c4dc87b40b95f75911938897cf8`.
`continue-confound-final.xml`: **2 passed** for the two mixed-intervention
negatives. `continue-inventory.xml`: **4 passed** for the shared-source inventory
and unchanged ownership restrictions. Suites overlap and are not independent
scientific experiments. Earlier fixture/expectation failures remain on E;
the initial confound test expected a later error message although the original
comparator had already rejected the input, and its corrected assertion passed.
Raw JUnit/observations remain under `E:/MolGap/artifacts/reform`.

This was bounded **INFRA_CPU_ACCEPTED** evidence, not a live platform or native
qualification. No GPU submission, new model, dataset access, historical decision
rewrite or source regrouping was performed. Desktop application, real platform
latency and overall A–G completion remain unestablished.

Primary integration of this follow-up fast-forwarded `c3f4c97b` to `cc7cbea3`.
All 14 pre-existing user byte pins matched before/after integration.
`continue-primary.xml` passed **44 tests** (input, recovery, original journal and
portable inventory; overlapping the suites above), zero failures/errors/skips.
The three changed operating documents passed navigation checks: 57 pointers,
one anchor, zero issues and no length gate. Core/Recipe/Kernel/Runner, package,
workflow/cache, RML and submission-journal owners had no diff from `c3f4c97b`.
No remote push or job operation was performed in this continuation follow-up.

## A-G completion re-audit and bounded follow-up (2026-10-17)

The reform plan was reread against server `23784af0`, not treated as an
implementation report. A read-only fetch observed remote Desktop
`1ebb2e45d78d623374b50fb5df40b7af8e51ce59`; the local Desktop ref was the
different `f4956019453b75d85347038d46a917fddc9dbdd1`. Neither ref was updated.
Core/state paths matched the remote snapshot, but five data/kernel/loader/recipe/
session owners and catalog glue differed. File-name matching is insufficient
for applying the shared patch, and local/remote Desktop refs are not substitutes.

The user assigned parallel implementation to GPT-6.1 Sol/high workers and
test execution to Luna workers; the coordinator did not execute tests. Disjoint
workers implemented bounded artifact metadata census, original-run lifecycle
resolution, timing evidence reporting and documentation convergence. No new
training loop, optimizer, sampler, platform submitter, RML store or scientific
contract was introduced.

- B: `artifact_inventory.py` counts explicitly selected regular-file metadata
  with entry/depth/time/Git-output limits, bulk Git classification and no link/
  reparse traversal or payload reading. Whole-worktree dirty state, unique
  copies and process custody remain unknown; partial scans are not cleanup
  evidence. Copying/verifying the earlier archive did not authorize deletion.
- E: the existing inspection, acceptance and resume CLI operations resolve the
  original frozen Spec/package/receipt from run/attempt IDs. Missing launches,
  ambiguous journals and changed source/preparation bindings fail closed.
  Scientific expectations, retrieved outputs, locations and recovery plans
  remain explicit inputs to their original owners. There is no inferred verdict
  or automatic successor. The legacy `--spec` arguments remain supported.
  The existing Kaggle family retriever gained a thin run-ID entry: local mode
  checks the exact original context and independently pinned manifest without
  credentials, network or destination creation. Explicit execution delegates
  to the original authenticator and pinned-file transport. Its output-list
  latest-session limitation is retained, not masked as version-specific listing.
- Performance evidence: successful preparation calls retain immutable local
  observations outside frozen input/prepared bytes and the operation journal.
  Auxiliary observation write failure is reported separately from the durable
  package. `experiment_timings.py` reads bounded allowlisted metadata, separates
  active phase accumulators from wall waiting, and reports missing duration/cost
  as missing. Descriptive sample quantiles do not establish stable population
  latency, warm applicability or the five-minute live target.
- F: five workflow/API guides lost 187 net lines after the retrieval pointer
  was integrated (1,216 to 1,029 lines), removing duplicate procedures and
  manual source-registration instructions. Two linked examples were corrected
  to declare ordinary addon execution once in the catalog. The quickstart is
  the sole normal workflow; detailed APIs and old package/readers remain.

### Completion boundaries

A is locally accepted. B covers new external storage and the bounded census
tool, not verified historical cleanup or complete copy uniqueness. C satisfies
the plan's first shared K1/GPTrans parameter/addon path; unsupported recipe,
input and new-family changes still require their actual owning builders, not
an arbitrary override compiler. D retains verified fixed distribution/cache
and separate Spec/package identities; a seed change that also changes recipe
bytes conservatively misses the archive cache. No new config/archive layer was
invented to hide that limit.

E's CPU/fake-platform path cannot establish authenticated live latency or
remote qualification. F's server convergence is not Desktop application: the
owner must review explicit shared commits against its actual checkout and
preserve its unique recovery/execution behavior. G stays deferred until this
cross-owner gate, as required by plan sections 14–15; no mass source move,
historical class-path deletion, worktree removal or hot environment update
was performed. Overall A–G completion remains **NOT_COMPLETE**.

### Retained follow-up test evidence

Raw reports remain under `E:/MolGap/artifacts/reform`; counts overlap and are
not independent experiments. The coordinator only read results; Luna workers
executed the tests. Earlier red reports were retained, not xfailed or replaced
by passing scientific claims.

| Report suffix (after `completion-`) | Observed result | Scope |
|---|---|---|
| `input-20261017-020156-753-b6dd5a0b.xml` | 37 passed, 20.003s | Original prepare/retry and retained-observation behavior |
| `input-20261017-020242-641-cace8bb8.xml` | 5 passed, 1.717s | Damaged original binding rejected |
| `run-20261016T170738953Z-48138d309556468e92177591054c11d1.xml` | 149 passed, 2 failed, 208.097s | 59 run-ID, 79 CLI, 3 original fake-journal cases passed; two resume fixtures failed before runtime assertions |
| `inventory-20261017-021335-607-1369091c.xml` | 73 passed, 3 failed, 2 skipped, 24.565s | Timing, package and environment checks passed; census Git classification failed |
| `inventory-fix-20261017-021634-131-3563c318.xml` | 21 passed, 1 failed, 1 skipped, 6.473s | Census Git fix; remaining deterministic fixture-byte mismatch |
| `inventory-bytefix-20261017-021855-765-74c2206a.xml` | 1 passed, 1.345s | Fixed-byte census fixture passed |
| `retrieval-run-20261016T171359425Z-1b84dcc77eb947c780e1bdb81857d7c3.xml` | 20 passed, 25 failed, 1 skipped, 78.661s | Both corrected resume cases and original retrievers passed; new retrieval fixture used the wrong platform |
| `reform-retrieval-run-20261016T171751557Z-0fa28d861b8341cbbba53dfde73e5f73.xml` | 33 passed, 1 skipped, 48.732s | Corrected actual Kaggle-context fixture, transport and negative boundary cases |

The three fixes preserve the failing assertions and production gates:
`check-ignore --stdin` rejects Git's global literal-pathspec flag, so only that
pathname-consuming command omits it; the census fixture writes known LF bytes
instead of assuming Windows `write_text` preserves LF; the synthetic execution
registry overrides only its selected two owners rather than deleting unrelated
families required by lazy imports. The retrieval fixture binds Kaggle before
building its real synthetic package/receipt; local-platform and abbreviated
`--exe` requests remain rejected. Scientific runtime owners were not changed.

One first Luna execution attempted a restricted nested environment and produced
23 setup permission errors, not business assertion failures. Its original JUnit
was retained at `E:/MolGap/scratch/reform-completion/retained-c-sandbox-failure-1792170409147-d55fd9ecfda9432cb69b466bb6badb9c/completion-storage.xml`,
SHA256 `9787e51b7b6d9124b8befb2582ad403f361eb00b23ab265ebd9d667a78b1a8ff`.
The transient C-drive test directory was moved there with original evidence
preserved; no project artifact or worktree was deleted. The later module tests
ran directly in Luna's permitted E-drive environment. Native symlink-creation
skips remain host limitations, with mocked reparse coverage separate.

Implementation commits are independently applicable: `066749d2` (B census),
`ec4675df` (preparation observations/report), `4d20c5d3` (E original-run
lifecycle/retrieval), and `505fcd0c` (D host-only distribution boundary and the
52-file portable inventory). Desktop must review these alongside the earlier
compiler/continuation patches; neither matching names nor this inventory is a
certificate of Desktop installation. No push or remote job operation followed.

### Bounded physical census observation

Luna inspected metadata only for the explicit old source-archive directory and
five registered worktrees. The source-archive directory contained three ignored
regular files totaling 28,189 logical bytes; four entries were visited in 0.250s.
Its complete small-scope report is
`E:/MolGap/artifacts/reform/completion-sourcearchive-census-20261017-022311-422-774a688c.json`,
SHA256 `9496e14efef23892e3e9d824b96d4c73effde1602b12126f7ab3602173b0cab9`.

The five-root scan was explicitly partial: 42,187 entries, 33,584 regular files
and 29,279,656,390 logical bytes were observed, with Git classification unknown.
The combined limit was 50,000 entries, depth12 and 30s; active return took
30.593s including cancellation/return overhead. Depth/time limits, excluded
nested worktrees and incomplete Git classification prevented a full inventory.
The report is
`E:/MolGap/artifacts/reform/completion-worktree-census-20261017-022311-422-774a688c.json`,
SHA256 `f916f8efdfb41a83fbeef1e9cc7ce32346260c3bcb07ff74921f5fc39308a6af`.
These are observed lower bounds, not total disk allocation or reclaimable space.
Payload contents were not opened or hashed. Unique-copy custody and process
dependencies remain unknown; nothing was cleared or declared safe to delete.
The earlier interrupted zero-byte report was retained and is not evidence.

### Shared inventory and navigation acceptance

Luna passed 17 focused checks: four portable-inventory checks, four reuse-skill
checks and nine documentation-check tests. The 52-file shared inventory matched;
native qualification and Desktop application remained explicitly unestablished.
Eleven changed active documents had 304 checked pointers, 18 checked anchors and
zero issues, without a line-count gate. An incremental check of the later
cumulative migration text had 11 pointers and zero issues. The other ten
documents and tested source/inventory bytes had not changed between those
observations. All 14 original user byte pins matched independently.

The final metadata at
`E:/molgap-leaf-testing-c00d612cedcb4c8a958f632c1574205c/result_final.json`
pins each JUnit, raw log and navigation result; SHA256
`289e0c2755249641a0f8050b506403678a62ac1819d5253654829bd1f6185e78`.
These tests ran in parallel leaf processes with separate E-drive scratch;
they are CPU/local checks, not live receipts or scientific tests.

During the final isolated CPU check, closing an unresponsive Luna conversation
did not terminate its already launched test process. A replacement invocation
therefore briefly duplicated the local verification. The later process tree
(verified root PID24428, scratch suffix `6ec44c43d7f1492cb47a455c39568c5c`)
was explicitly stopped; its partial log/scratch was retained, not counted as a
passing suite. The first process (scratch suffix
`b29a52e8-7ad0-4b98-ae32-625f48c194c7`) was left to finish. This affected only
CPU synthetic verification; no remote run or source package was resubmitted.
Conversation status alone is not proof that its background command stopped.

### Final committed-distribution and scientific-entry CPU acceptance

At source `6cc71584b348e5843de1ebcddcfe9dbbad457ebb` (source unchanged from
`505fcd0c`), Luna's original invocation passed **21 tests**, zero failures/errors/
skips, in 242.684s JUnit time (244.275s wrapper wall time). It exercised the
actual fixed distribution and complete K1/GPTrans scientific-input/addon,
pre-publication interruption, changed-reference, cross-HEAD continuation and
confounded-intervention CPU scenarios. Both source/HEAD snapshots were identical
before and after. The JUnit is
`E:/MolGap/artifacts/reform/completion-distribution-b29a52e8-7ad0-4b98-ae32-625f48c194c7.xml`,
SHA256 `daa04d309d2b83e22cb19eae213c07d0cb1e989c22d072add165b2a2287da638`.
The raw command/snapshots/manifests/measurements are in the adjacent `.raw.json`,
SHA256 `0c05e96233de5af0fa52622f53095e7a2a7fc60fa955e73536708acb751e5cdd`.
The six pytest warnings concern `record_property` with JUnit xunit2, not hidden
test failures. Raw reports retain the properties and original observations.

The 426-file distribution archive contained 1,304,088 bytes. Cold/warm source
package samples took 5.062498/3.671561s; the isolated CPU child took 19.036014s.
K1 and GPTrans sandbox checks passed with all MolGap modules imported from the
extracted E-drive package, not host source. Existing Core/Session/State tests
also ran inside that child: 116 passed in 5.695s, zero failures/errors/skips;
the retained `driver/cpu-state.xml` SHA256 is
`373e7e71ea359051195c00555d7aa2f9a7bc5edb11e7e255f8b6e09c5afffd59`.
These nested tests overlap earlier module regressions. The package identities differed while source
archive bytes were reused; no receipt or scientific identity was borrowed.
These are raw local samples, not a stable latency distribution, a 90-second
preparation certificate, a live five-minute receipt, or native qualification.

Final delivered scope: **INFRA_CPU_ACCEPTED for the server subset**.
Overall reform remains **NOT_COMPLETE**: Desktop application/compatibility
requires the independent owner's actual checkout, G source regrouping follows
that gate, historical unique-copy recovery remains unproved, and authenticated
platform latency/native qualification were not run. All 14 unrelated user byte
pins remained identical; no historical scientific conclusion, V5 text, training
core, recipe, optimizer, sampler, resume owner, platform submission journal or
canonical RML implementation was changed. No remote push, GPU job, dataset
rebuild, protected-role access, model promotion or worktree cleanup occurred.

## Server-only source regrouping and recovery (2026-10-17)

The user subsequently authorized the remaining reform on `molgap-server`,
explicitly excluding Desktop synchronization. This changed the earlier G
deferral: the server may organize its own implementation while keeping a
reviewable portable patch. It did not authorize training, protected roles or
rewriting historical contracts. The earlier dated results above are unchanged.

### Replaced paths and retained compatibility

Five GPT-6.1 Sol/high implementation workers had disjoint ownership. They
relocated existing implementations rather than creating new training/data
systems; the coordinator integrated packaging and public-import boundaries.

| Existing responsibility | Implementation owner | Preserved reader |
|---|---|---|
| Family loop, Session and complete state | `training/core.py`, `training/session.py`, `training/state.py` | Old `family_training_*` modules alias the same module objects |
| Substrate slots, attachment and parameter ownership | `models/substrate.py` | `model_substrate.py` aliases it |
| Packed graphs, fixed manifest records and bounded indexed loading | `data/packed_graph.py`, `data/fixed_records.py`, `data/indexed_inputs.py` | Old module objects, private monkeypatch paths and pickle names |
| Nine historical-only constants | `compat/legacy_constants.py` | Lazy `constants` and package exports, including old star imports |
| Frozen GPTrans implementation | Original `gptrans.py` | `models/gptrans.py` aliases the original implementation |

The GPTrans exception is deliberate: `pcqm_gptrans_v4.py` hashes the literal
historical file, not merely equivalent equations. Its raw SHA256 remains
`04edcb6f928617d1142c624d071b7d7accb15c92d2f66e3473ed666ca7e5b2b2`.
No expected hash or scientific identity was relaxed to accommodate a move.
The LF attribute protects these exact bytes on subsequent checkouts.

Historical type/function `__module__` names are retained for pickles. Old/new
imports share one implementation and globals, rather than maintaining copies
or hiding private monkeypatches behind star imports. Package imports remain
lazy. Live model registry, asset locations and inference defaults are unchanged.
Runtime source inventories and the older CPU cycle stage include the new
implementation closure; RML portability includes the lazy compatibility owner.
The portable reform inventory contains 67 normalized-source pins. Matching it
does not imply Desktop installation or native qualification.

### Source archive reuse without identity reuse

The existing six-file package and verifier were retained. Source cache keys now
cover builder version, exact selected paths and normalized committed payload
digests instead of the current commit string. An unrelated documentation commit
can reuse the archive bytes; its package receives fresh commit/inventory
sidecars and a distinct package identity. Original cached metadata is checked
before rebinding; the ordinary verifier rechecks all actual archived payloads.
Dirty or filter-hidden source, corrupt archives/sidecars and a changed HEAD
before publication fail closed. There is no reuse of another Spec, receipt or
scientific acceptance.

The cache's creation provenance is retained unchanged. V1/V2 package readers and
old cache objects remain intact; the versioned new key causes one cold miss
when transitioning from commit-keyed caches. Identical payloads with a
seed-only synthetic Spec reuse the source bytes, but changed included recipe
bytes intentionally rebuild the full archive. That is not a promise of archive
reuse when actual packaged contents differ.

### Verified physical worktree retirement

Only this chat's ended `codex/reform` worktree at
`C:/Users/Adminn/.codex/worktrees/modern2d-cpu/molgap` was retired. Its HEAD was
`23784af09e0b339c52814641db28ceb8ea393719`; nonignored working files were clean.
Before retirement, the editable environment was checked against its actual
`.pth` target and process command lines were checked for dependencies on this
exact worktree. Neither check authorized changing the environment itself.

All 733 ignored regular files (12,468,914 logical bytes) were stream-copied and
hash-verified under
`E:/MolGap/artifacts/reform/retained-worktree-modern2d-cpu-23784af0/ignored/`.
The immutable `ignored_files_manifest.json` SHA256 is
`2ec4f3438d4f99c3fedb77a2f5ac21f2714eda49a74be5d899d1b3cb585e017a`.
It describes the pre-retirement copy observation, when originals still existed;
its original-retained flags were not rewritten after cleanup. A verified
complete-history `source-history.bundle` preserves the old branch at the same
E-drive root (152,191,128 bytes; SHA256
`62bcd012a420df6fcc1146dab39f456dfcb3f9be52bd0f37e58da882652192c7`).

The app's recoverable worktree archive completed: the attachment became
`archived_worktree`, and Git's registered physical worktrees fell from five to
four. The retained source bundle and ignored files provide an additional
independent recovery route. The main checkout, archive-branch worktree,
other-owner detached checkout and frozen staging checkout were not removed.
Logical byte counts are not measured reclaimed filesystem allocation. This is
one exact safe retirement, not a claim that every historical copy is redundant.

### CPU evidence and completion scope

Luna leaf workers executed all tests directly with separate E-drive scratch and
reports, hidden CUDA and single-threaded CPU kernels. The main agent executed
no tests. The raw reports below overlap; they are not independent experiments.

| Report under `E:/MolGap/artifacts/reform/` | Result | Scope |
|---|---|---|
| `server-completion-training-43b5789f-9c8e-4c1a-ae90-4b32113b4639.xml` | 142 passed, 16.380s | Frozen AST parity, module/pickle aliases and existing loop/Session/state including interruption and recovery |
| `server-completion-models-b818528215034a4f8f4b906abfce1d44.xml` | 78 passed, 68.410s | Model migration, substrate slots, sandbox and topology reuse before GPTrans's byte-pin exception |
| `server-completion-models-18cc4af07052421e8fe9f04fe64b690e.xml` | 23 passed, 35.190s | Final model namespace and unchanged GPTrans V4 initial parameter/source identity |
| `server-completion-data-6fe5ecd2d70742af9e53deab7c76418e.xml` | 52 passed, 53.991s | Data aliases/ABI/indexed negative cases and all historical constants/inference defaults |
| `server-completion-cacheworker-62c20e41-237a-4421-9c61-ea203eed486a.xml` | 89 passed, 1 skipped, 135.909s | Package/cache, fresh cross-commit sidecars, retry and corruption negatives |

The cache skip is native symlink creation unavailable on this host; it is not a
passing symlink capability. The separate portability JUnit at
`E:/MolGap/artifacts/reports/portability-9c8bc9fb23e4417b82d9b7a90ccfc233.xml`
passed two cases in 2.552s (SHA256
`b5c8313bb3faab7c19bc10336dd1c1f236cfce38e2df18f8ec343d4972cf0b9c`).

Implementation commits are `712146e9` (G owners/compatibility and runtime
closure) and `5bfc5ab2` (D verified content cache). Final isolated-distribution,
scientific-input and portable-inventory evidence is recorded below after the
committed-source check; no pending check is described as accepted here.

Desktop remains `NOT_APPLIED` by explicit user scope, not a server code blocker.
Authenticated upload/source-ready/receipt latency, accelerator qualification,
arbitrary recipe overrides and unexamined historical storage remain outside
the CPU evidence. Existing native gates and unsupported-family limits remain
unchanged. No remote job, dataset rebuild, protected-role access, promotion,
shared-environment rewrite or bulk historical deletion occurred.

### Final integration observations

The first portable inventory check had seven passes and one sorting failure:
all 67 unique files matched their normalized hashes, but PowerShell's cultural
ordering put underscores before directory separators. The inventory was sorted
by ordinal POSIX paths without changing any hash. The targeted rerun passed all
four compatibility cases in 0.419s; the four reuse-skill cases were not rerun.
Both reports remain at `E:/MolGap/artifacts/reports/`:
`shared-reuse-2c762aed8a2448668aaaedec62f0e210.xml` (original red report) and
`shared-compat-abaa81ccc40342a9935433b73b43682a.xml` (green SHA256
`6ca74c3ac8251452720c3e0915e9eb76e96185427b3c9c59317f292f7448838f`).
The 14 pre-existing user file pins also matched raw bytes, including the dirty
RML-derived outputs, experiment READMEs and `tests/test_research_memory.py`.
They were not staged, overwritten or made part of these commits.

The first final-source invocation at `5bfc5ab2` passed all 20 scientific-input
cases in its 181.31s run, but its separate actual-distribution case was blocked
before publishing a package by the existing clean-source gate. Adding the LF
attribute left Git's old stat entry flagging `gptrans.py` despite identical raw
bytes/blob identity; staging that exact unchanged file refreshed the index,
with no content change or commit. The frozen SHA gate was not bypassed.
The original failed report/log were retained under
`E:/MolGap/artifacts/reform/server-completion-final-distribution-18658996-5cb3-47a7-ba81-dd634e31e882.*`.
Only the blocked distribution case was selected for a fresh check after the
source/index were confirmed clean; the 20 successful cases were not repeated.

The four edited navigation documents had 319 checked pointers, eight anchors
and zero issues; three focused documentation-owner tests passed in 0.45s.
That report precedes this final appendix, which receives an incremental check.
The complete earlier navigation report is
`E:/MolGap/artifacts/reports/docs-owner-0098f78971e545aa901155798144d27d.json`,
SHA256 `4fd087e4aa4159d96abe8dfa185f9c7ff32a9fc2af82e6bf9568063fbab967de`.

The single distribution rerun passed in 30.424s JUnit time at unchanged
`5bfc5ab2f385587a5ec06c150ccd96c5a90ff804`. Its independently extracted
439-file archive contained 1,307,066 bytes. Cold/warm package samples were
5.469637/3.787575s; the isolated CPU child took 16.836722s. Both K1 and GPTrans
addon sandbox checks passed, and the original Core/Session/State suites passed
all 116 cases inside that extracted package. Every loaded MolGap module came
from the package, not the host checkout. Cold/warm packages reused identical
archive bytes while retaining distinct package identities. These are individual
local samples, not p50/p95 estimates or proof of five-minute remote receipts.

The green outer JUnit is
`E:/MolGap/artifacts/reform/server-completion-final-distribution-rerun-bcf54746-d58f-4088-9b36-dcc626cacabd.xml`,
SHA256 `6be7afbeb82bd01c8b42b283b378fcb026b672803196dc5c2b1ca314091f31b1`.
The extracted-package nested JUnit is retained below the corresponding E-drive
scratch at `test_committed_distribution_im0/driver/cpu-state.xml`, SHA256
`719509b03bfd38cd0c682493b8c9ff9b225a19854a2dd7e430adf205952ff5f2`.
The initial 20-pass/one-blocked JUnit SHA256 is
`e0eb00f123a3e1d778125e1a551cac80b68bfc66976f023fec2398331945d754`;
neither that evidence nor either earlier red report was discarded or xfailed.

Delivered status: **INFRA_CPU_ACCEPTED for the server-local reform scope**.
A/new-task B, the supported K1/GPTrans C-D path, existing E journal/lifecycle,
server documentation F and bounded active-owner G now have CPU evidence.
This task did not add a second core, catalog, RML database, submitter or generic
cleanup service. G is responsibility-based relocation, not a claim of net
source-line deletion; the earlier duplicate workflow removal remains separately
measured above. One ended physical checkout was actually retired recoverably.

Remaining boundaries are explicit: Desktop synchronization was excluded; live
authenticated latency has **INSUFFICIENT_EVIDENCE**; unexamined historical
directories are not declared safe to delete. Source-pinned historical paths,
registry-backed compatibility constants and unsupported scientific recipe/input
changes remain with their owning contracts. None is silently promoted to a
supported arbitrary override. New native/model/data qualification is separate
from this CPU reform acceptance. The implementation commits and final compact
records are local to `molgap-server`; no push was performed in this task.
